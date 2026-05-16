import re
from fractions import Fraction


BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")


def strip_solution_text(text: object) -> str:
    if text is None:
        return ""
    return str(text).strip()


def extract_answer(text: object) -> str:
    """Extract a math final answer from common LLM output formats."""
    s = strip_solution_text(text)
    if not s:
        return ""

    boxed = BOXED_RE.findall(s)
    if boxed:
        return normalize_answer(boxed[-1])

    if "\\frac" in s:
        frac = re.findall(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", s)
        if frac:
            num, den = frac[-1]
            return normalize_answer(f"{num}/{den}")

    lower = s.lower()
    markers = ["final answer is", "answer is", "therefore", "so the answer is"]
    for marker in markers:
        idx = lower.rfind(marker)
        if idx != -1:
            tail = s[idx + len(marker) :]
            nums = NUMBER_RE.findall(tail)
            if nums:
                return normalize_answer(nums[-1])

    nums = NUMBER_RE.findall(s)
    if nums:
        return normalize_answer(nums[-1])
    return normalize_answer(s)


def normalize_answer(text: object) -> str:
    s = strip_solution_text(text)
    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.strip()

    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    if s.endswith("."):
        s = s[:-1].strip()

    frac = re.fullmatch(r"\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", s)
    if frac:
        return str(Fraction(int(frac.group(1)), int(frac.group(2))))

    try:
        if "/" in s and re.fullmatch(r"[-+]?\d+/[-+]?\d+", s):
            return str(Fraction(s))
        if re.fullmatch(r"[-+]?\d+\.0+", s):
            return str(int(float(s)))
    except Exception:
        pass

    return s.lower().replace(" ", "")


def is_correct(prediction: object, reference: object) -> bool:
    pred = extract_answer(prediction)
    ref = extract_answer(reference)
    if pred == ref:
        return True

    try:
        return Fraction(pred) == Fraction(ref)
    except Exception:
        return False
