"""Answer extraction and checking, copied from the authors' exp8_math/code/mathqa.py (MIT, see
third_party/lm_understand_logs/LICENSE) so that our scores use exactly their rule: the last \\boxed{} after the last
</think> (the whole text when there is no </think>), checked with math-verify 0.9.0.

Also the loop detector used by the loop-break decoding arm (a5): the authors' verbatim_loop_end rule plus a
repeated-4-gram rule of our own (their mcqa.repetitive_end is not published).
"""
import functools
import re
import threading

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(problem):
    return problem + "\n" + INSTRUCTION


def user_prompt(row):
    """the user turn of a row: multiple-choice rows carry their own prompt (wsa.prepare_med), math rows get the
    authors' instruction."""
    return row["prompt"] if row.get("kind") else build_prompt(row["problem"])


def answer_text(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def _box_at(text, idx):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text):
    t = answer_text(text)
    return last_boxed(t), len(_BOX_FORMS.findall(t))


def _timeouts():
    return 5 if threading.current_thread() is threading.main_thread() else None


@functools.lru_cache(maxsize=1 << 16)
def _parsed(s, to):
    import math_verify
    return math_verify.parse("\\boxed{" + s + "}", parsing_timeout=to)


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    to = _timeouts()
    g, p = _parsed(gold, to), _parsed(pred, to)
    if not g or not p:
        return False
    try:
        import math_verify
        return bool(math_verify.verify(g, p, timeout_seconds=to))
    except Exception:  # noqa: BLE001
        return False


def mc_extract(text):
    """multiple-choice letter, the official MMLU-Pro reading (TIGER-Lab evaluate_from_api.py: "answer is (X)", else
    "Answer: X", else the last standalone letter A-J) on the text after the last </think>, as the authors read
    MMLU-Pro and GPQA."""
    t = answer_text(text)
    m = re.search(r"answer is \(?([A-J])\)?", t)
    if m:
        return m.group(1)
    # ".*[aA]nswer:\s*([A-J])" (no DOTALL) picks, in the first line that has one, the last place where "answer:" is
    # followed by a letter. The regex itself takes time quadratic in the line's length and never finished on the
    # multi-megabyte lines some looping answers have, so the same choice is made here in one pass.
    best = None
    for m in re.finditer(r"(?=[aA]nswer:\s*([A-J]))", t):
        line = t.rfind("\n", 0, m.start())
        if best is not None and line != best[0]:
            break
        best = (line, m.group(1))
    if best is not None:
        return best[1]
    # "\b[A-J]\b(?!.*\b[A-J]\b)" with DOTALL is the last standalone letter (same quadratic cost)
    last = None
    for last in re.finditer(r"\b[A-J]\b", t):
        pass
    return last.group(0) if last else None


def grade(row, text):
    """(prediction, correct) of one answer: multiple-choice rows by letter, math rows by the authors' boxed rule."""
    if row.get("kind") == "mc":
        pred = mc_extract(text)
        return pred, pred == row["answer"]
    pred, _ = extract(text)
    return pred, is_correct(pred, row["answer"])


# ---------------------------------------------------------------- loop detection (a5)
LOOP_MAX_PERIOD, LOOP_MIN_REPEATS, LOOP_MIN_SPAN = 200, 3, 40


def verbatim_loop_end(words):
    """the authors' rule: the text (minus its last word) ends with one block of 1..200 words repeated in a row at least
    3 times and over at least 40 words."""
    w = words[:-1]
    n = len(w)
    for p in range(1, LOOP_MAX_PERIOD + 1):
        reps = max(LOOP_MIN_REPEATS, -(-LOOP_MIN_SPAN // p))
        if reps * p > n:
            break
        blk = w[n - p:]
        if all(w[n - (j + 1) * p:n - j * p] == blk for j in range(1, reps)):
            return True
    return False


def repetitive_end(words, window=200, share=0.5):
    """in the last `window` words, the 4-word positions whose phrase occurs 3+ times there are at least `share` of all
    positions (circling with small variations)."""
    w = words[-window:]
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    if len(grams) < window // 2:
        return False
    cnt = {}
    for g in grams:
        cnt[g] = cnt.get(g, 0) + 1
    return sum(1 for g in grams if cnt[g] >= 3) >= share * len(grams)


TAIL_CHARS = 12000      # both rules read at most the last 600 words; callers may pass only the text's end


def looping(text):
    words = text.split()
    return verbatim_loop_end(words) or repetitive_end(words)
