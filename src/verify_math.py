"""Deterministic GSM8K and conservative MATH answer verification."""

from __future__ import annotations

import contextlib
import re
import signal
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Iterator


NUMBER_RE = re.compile(
    r"(?<![\w.])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?%?)"
)
LATEX_FRACTION_RE = re.compile(
    r"\\(?:d)?frac\{\s*([+-]?(?:\d+(?:\.\d+)?))\s*\}"
    r"\{\s*([+-]?(?:\d+(?:\.\d+)?))\s*\}"
)
FINAL_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|therefore)[^\n:=]*[:=]?\s*([^\n]+)",
    flags=re.IGNORECASE,
)


class VerificationTimeout(RuntimeError):
    pass


def extract_last_boxed(text: str) -> str | None:
    position = text.rfind(r"\boxed")
    if position < 0:
        return None
    opening = text.find("{", position)
    if opening < 0:
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index].strip()
    return None


def extract_final_answer(text: str) -> str | None:
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed
    if "####" in text:
        candidate = text.rsplit("####", maxsplit=1)[-1].strip().splitlines()[0]
        if candidate:
            return candidate
    matches = list(FINAL_RE.finditer(text))
    if matches:
        candidate = matches[-1].group(1).strip().rstrip(". ")
        if candidate:
            return candidate
    numbers = NUMBER_RE.findall(text)
    return numbers[-1].strip() if numbers else None


def normalize_gsm8k(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = value.strip().strip("$ ").rstrip(". ")
    latex_fraction = LATEX_FRACTION_RE.search(candidate)
    if latex_fraction:
        token = f"{latex_fraction.group(1)}/{latex_fraction.group(2)}"
    else:
        match = NUMBER_RE.search(candidate)
        if not match:
            return None
        token = match.group(1)
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    token = token.replace(",", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", maxsplit=1)
            fraction = Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        else:
            fraction = Fraction(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    normalized = str(fraction.numerator)
    if fraction.denominator != 1:
        normalized += f"/{fraction.denominator}"
    return f"percent:{normalized}" if is_percent else normalized


@contextlib.contextmanager
def _time_limit(seconds: int) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(_signum: int, _frame: Any) -> None:
        raise VerificationTimeout(f"verification exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def verify_completion(
    verifier: str, completion: str, gold_answer: str, timeout_seconds: int = 10
) -> dict[str, Any]:
    extracted = extract_final_answer(completion)
    result: dict[str, Any] = {
        "parsed": False,
        "correct": False,
        "extracted_answer": extracted,
        "normalized_prediction": None,
        "normalized_gold": None,
        "failure_reason": None,
    }
    if extracted is None:
        result["failure_reason"] = "no_final_answer"
        return result

    if verifier == "gsm8k":
        prediction = normalize_gsm8k(extracted)
        gold = normalize_gsm8k(gold_answer)
        result.update(normalized_prediction=prediction, normalized_gold=gold)
        if prediction is None:
            result["failure_reason"] = "prediction_parse_error"
            return result
        if gold is None:
            result["failure_reason"] = "gold_parse_error"
            return result
        result["parsed"] = True
        result["correct"] = prediction == gold
        if not result["correct"]:
            result["failure_reason"] = "not_equivalent"
        return result

    if verifier != "math":
        raise ValueError(f"Unknown verifier: {verifier}")

    result.update(normalized_prediction=extracted, normalized_gold=gold_answer.strip())
    try:
        from math_verify import parse, verify

        with _time_limit(timeout_seconds):
            parsed_gold = parse(gold_answer)
            parsed_prediction = parse(completion)
            if not parsed_prediction:
                parsed_prediction = parse(extracted)
            if not parsed_gold or not parsed_prediction:
                result["failure_reason"] = "symbolic_parse_error"
                return result
            result["parsed"] = True
            result["correct"] = bool(verify(parsed_gold, parsed_prediction))
            if not result["correct"]:
                result["failure_reason"] = "not_equivalent"
    except VerificationTimeout:
        result["failure_reason"] = "timeout"
    except Exception as exc:  # malformed model output is incorrect, but auditable
        result["failure_reason"] = f"verifier_exception:{type(exc).__name__}"
    return result
