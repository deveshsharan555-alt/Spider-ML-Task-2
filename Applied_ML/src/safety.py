"""
Safety and moderation layer.

"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SafetyFlag(Enum):
    OK = "ok"
    EMERGENCY = "emergency"
    UNSAFE_DOSAGE_OR_DIAGNOSIS = "unsafe_dosage_or_diagnosis"
    SELF_HARM = "self_harm"


@dataclass
class SafetyResult:
    flag: SafetyFlag
    message: str | None = None  # pre-written response to show instead of the RAG pipeline


EMERGENCY_PATTERNS = [
    r"\bchest pain\b", r"\bcan'?t breathe\b", r"\bnot breathing\b",
    r"\bsevere bleeding\b", r"\bheart attack\b.*\b(now|happening|right now)\b",
    r"\bstroke\b.*\b(now|happening|right now|symptoms right now)\b",
    r"\bface (is )?drooping\b", r"\bslurred speech\b",
    r"\bunconscious\b", r"\bnot responsive\b", r"\bturning blue\b",
    r"\boverdos(e|ed|ing)\b", r"\banaphyla", r"\bsevere allergic reaction\b",
    r"\bseizure happening\b", r"\bpassed out\b",
]

SELF_HARM_PATTERNS = [
    r"\bkill myself\b", r"\bsuicid", r"\bend my life\b",
    r"\bself[- ]harm\b", r"\bhurt myself\b", r"\bwant to die\b",
]

DOSAGE_DIAGNOSIS_PATTERNS = [
    r"\bhow many (pills|mg|milligrams)\b", r"\bwhat dose\b", r"\bwhat dosage\b",
    r"\bhow much .*(should i take|can i take)\b",
    r"\bdo i have (cancer|diabetes|a tumor|hiv|copd)\b",
    r"\bam i having a heart attack\b",
    r"\bwhat('s| is) wrong with me\b",
    r"\bcan i (stop|skip|double) (taking )?my medication\b",
    r"\bmix .*(with alcohol|and alcohol)\b",
]


def _matches_any(patterns, text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


EMERGENCY_MESSAGE = (
    "⚠️ What you're describing may be a medical emergency. "
    "This assistant cannot provide emergency care. "
    "Please call your local emergency number (e.g. 911 in the US, 999 in the UK, "
    "112 in the EU) or go to the nearest emergency department right now. "
    "If you can, ask someone nearby to stay with you until help arrives."
)

SELF_HARM_MESSAGE = (
    "I'm concerned about what you shared. You deserve immediate support from "
    "someone who can help right now. If you are in the US, you can call or text "
    "988 (Suicide & Crisis Lifeline), available 24/7. If you are elsewhere, "
    "please contact your local emergency number or a crisis line in your country. "
    "If you're not in immediate danger but this is weighing on you, please also "
    "consider reaching out to a trusted person or a mental health professional."
)

DOSAGE_DIAGNOSIS_MESSAGE = (
    "I can't provide personal dosing instructions, diagnose a condition, or tell you "
    "whether to change a prescribed medication — that requires a clinician who knows "
    "your medical history. I can share general, source-cited information about a "
    "condition or medication class instead. Please consult a doctor or pharmacist "
    "for anything specific to your situation."
)


def check_safety(query: str) -> SafetyResult:
    if _matches_any(SELF_HARM_PATTERNS, query):
        return SafetyResult(SafetyFlag.SELF_HARM, SELF_HARM_MESSAGE)
    if _matches_any(EMERGENCY_PATTERNS, query):
        return SafetyResult(SafetyFlag.EMERGENCY, EMERGENCY_MESSAGE)
    if _matches_any(DOSAGE_DIAGNOSIS_PATTERNS, query):
        return SafetyResult(SafetyFlag.UNSAFE_DOSAGE_OR_DIAGNOSIS, DOSAGE_DIAGNOSIS_MESSAGE)
    return SafetyResult(SafetyFlag.OK)


GENERAL_DISCLAIMER = (
    "This is general health information grounded in the sources cited below, "
    "not a diagnosis or personal medical advice. Please consult a qualified "
    "healthcare provider about your specific situation."
)
