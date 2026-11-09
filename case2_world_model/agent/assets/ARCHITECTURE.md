# The world model

An event-stream world model of ICU stays with 4.1 million parameters, width d = 192. It was trained by maximum
likelihood on the training stays (/data/train.csv) to predict each next group of events from everything before it.
Tensor names and shapes are in `tensors.json`. Codes, normalisation and bin tables are in `vocab.json`. Sizes are in
`vocab.json["config"]`.

## Input

A stay is a sequence of **bundles**. A bundle holds all events recorded in the same minute, and each event is an
**atom**. An atom has:
- a code (`vocab.json["codes"]`): glucose_finger, glucose_lab, map_arterial, map_cuff, insulin, norepinephrine,
  hospital_admission, icu_admission;
- an atom kind (0 WORLD; 2 END marks the end of an infusion record) and an event kind;
- its attributes: a measurement's value; a drug record's rate and dose (an infusion record runs for one hour at its rate).

**Embedding** (`embed.*`). Each atom's vector is the sum of these terms:
- code: `embed.code[code]`;
- kind: `embed.kind`, `embed.ekind`;
- unit: the `embed.categorical.*` tables;
- fine bins: `embed.bins.value[value bin]`, `embed.bins.dose[...]`, `embed.bins.rate[...]`;
- numbers: `embed.numeric @ [z_value, has_value, z_dose, has_dose, z_rate, has_rate] + embed.numeric.bias`.

Here z is a robust per-code score, and the fine bins (1..16, 0 = absent) come from the per-code edges in `vocab.json`.

**Set encoders** (DeepSets, same form): `norm(mlp2(sum_i mlp1(atom_i) / sqrt(n) + size(log(1 + n))))`.
- Measurements and visit events go through `encode.*`, giving the bundle vector e_k.
- Insulin and norepinephrine atoms are **actions** and go through `encode_action.*`, giving u_k (zero if the bundle has
  none).

## State and one step per bundle (`core.*`)

**State:**
- h: 4 slots x 192;
- a memory M of the last 128 bundle vectors e;
- `out` (192): the summary every head reads.

**One step for bundle k**, arriving dt seconds after the previous one:

1. **Time flow.** `h <- h * exp(-dt * exp(core.flow.a))`, a per-dimension decay with timescales from 10 minutes to 2 years.
2. **Latent z_k (32-dimensional).**
   - Context: `ctx = out + out_norm(mean_slots(h)) + gap(log dt / 10)`.
   - Prior: `z_prior(ctx)` gives mean and log-sd. Posterior: `z_post([ctx, e_k, u_k])`.
   - The state update uses the posterior mean.
3. **Prelude** (`core.prelude.0`, a pre-norm transformer layer).
   - It runs over the tokens [h + `core.slots` (4 tokens), e_k, u_k, `z_in(z_k)`].
   - The 4 slot outputs, through `core.prelude_norm`, give E.
4. **Recurrent block, one iteration** (`core.blocks.0.0`, `core.blocks.0.1`). Each layer has self-attention over the
   slots, cross-attention to M (keys get `core.age(log(1 + age))`) and an FFN.
   - `u = blocks(h + E)`.
   - `h <- a_bar * h + softplus(core.delta_in) * core.b_in(E) + (u - (h + E))`, with
     `a_bar = exp(-softplus(core.delta_in) * exp(core.a_in))`.
5. **Coda.** `y = coda(core.c_out(h))`. The new h is y, and `out = core.out_norm(mean_slots(y))`.

## Heads: predicting bundle k+1 from what is known before it

**The world's view:** `view = out + out_norm(mean_slots(h after the time flow)) + gap(log dt) + z_head(z)`.
- At evaluation, z is drawn from the prior 4 times and the probabilities are averaged.
- Actions are predicted from a filtered view: `filter_norm(view + sigmoid(filter_gate([view, p])) * pending_mlp(p))`.

**Heads:**
- **Gap:** `time_head` gives the time to the next bundle (a mixture of 4 log-normals).
- **Counts:** `count_head` (observations) and `count_head_action` (actions) give the number of atoms of each type
  (Poisson).
- **Code:** `q = query_norm(view + type_emb[type])`, and the logits are `q @ embed.code.weight.T + code_bias`.
- **Attributes.** They are read from:

      with_code = q + embed.code[code]
                  + last_numeric([has_last, log(1 + age) / 10, z_last_value, z_last_rate])
                  + embed.bins.value[last value bin] + embed.bins.rate[last rate bin] + last_status[...]

  The "last" fields are a register: the most recent earlier value of the same code in this stay.
  - **value:** `attr_heads.value_bin(with_code) + embed_transition.value_bin[code * 17 + last value bin]`, then a softmax
    over 17 classes (0 = no value, 1..16 = fine bins). The predicted mean is `sum_b p(b) * centre[b]`
    (`vocab.json["value_bin_centres"]`).
  - **rate:** `rate_bin` works the same way for drug rates.
  - **other heads:** `monitor_head` (a Gaussian on z), `status`, `exec*`, `delay_head` and `cont_head` (does the stay
    continue).

## How a dose edit is read

"Insulin x2 from hour 12":
1. The record up to ICU hour 12 is shown as it is.
2. From hour 13 on, readings are hidden. The model draws its own glucose and MAP values bundle by bundle and feeds them
   back in.
3. Every insulin atom from hour 12 on has its rate and dose doubled, which changes their z-scores and bins in the
   embedding. All other events are as recorded.
4. The readout is the predicted mean glucose_finger in each hour 13..24, edited minus factual. It is averaged over 20
   paths with common random numbers, at hours that come after the drug was running, then averaged over stays.

Norepinephrine edits are read on map_arterial in the same way.
