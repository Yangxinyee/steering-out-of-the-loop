# Training records

The records are from a simulated ICU. They are not patient data.

- `train.csv`: 30,000 stays. These are the stays the model was trained on.
- `tuning.csv`: 400 other stays from the same ICU.

Each row is one event. Times are hours since ICU admission (`icu_admission` is at hour 0).

| column | meaning |
|---|---|
| stay_id | one ICU stay |
| hour | when the event happened, in hours since ICU admission (events before it, such as the hospital admission, have negative hours) |
| event | glucose_finger, glucose_lab (mg/dL); map_arterial, map_cuff (mmHg); insulin, norepinephrine (infusion records); hospital_admission, icu_admission |
| value | the reading, for measurements |
| rate, rate_unit | the infusion rate: insulin in units/hour, norepinephrine in mcg/kg/min |
| dose, dose_unit | the amount given over the record (rate x 1 hour; insulin in units, norepinephrine in mg) |
| end_hour | when the infusion record ends (one hour after it starts), or when the stay ends for the admission rows |

Each infusion record is one hour at its rate. Clinicians chose the doses, as in any ICU record.
