"""Guardrails: PII detection/redaction (Presidio when available, regex fallback incl. India-specific
IDs), prompt-injection defense (direct + indirect/retrieved-content), and content filtering."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass
class Finding:
    kind: str
    start: int
    end: int
    text: str
    score: float = 0.9
    source: str = "regex"

    def to_dict(self):
        return asdict(self)


def _luhn(num: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", num)][::-1]
    if not 13 <= len(d) <= 19:
        return False
    return sum(x if i % 2 == 0 else (x * 2 - 9 if x * 2 > 9 else x * 2) for i, x in enumerate(d)) % 10 == 0


PII_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), 0.99),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"), 0.85),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), 0.95),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.9),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), 0.9),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)|\+\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}"), 0.8),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0.8),
    ("API_KEY", re.compile(r"\b(?:sk|pk|hf|ghp|AKIA)[-_A-Za-z0-9]{16,}\b"), 0.95),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), 0.9),
]

_presidio = None
_presidio_tried = False


def _get_presidio():
    global _presidio, _presidio_tried
    if _presidio_tried:
        return _presidio
    _presidio_tried = True
    try:
        from presidio_analyzer import AnalyzerEngine

        _presidio = AnalyzerEngine()
    except Exception:
        _presidio = None
    return _presidio


def detect_pii(text: str, use_presidio: bool = True) -> list[Finding]:
    found: list[Finding] = []
    for kind, pat, score in PII_PATTERNS:
        for m in pat.finditer(text):
            if kind == "CREDIT_CARD" and not _luhn(m.group()):
                continue
            if kind == "AADHAAR" and _luhn(m.group()) and len(re.sub(r"\D", "", m.group())) > 12:
                continue
            found.append(Finding(kind, m.start(), m.end(), m.group(), score))
    eng = _get_presidio() if use_presidio else None
    if eng:
        try:
            for r in eng.analyze(text=text, language="en", entities=["PERSON", "LOCATION", "NRP", "IBAN_CODE"], score_threshold=0.6):
                found.append(Finding(r.entity_type, r.start, r.end, text[r.start:r.end], float(r.score), "presidio-ner"))
        except Exception:
            pass
    # resolve overlaps: keep highest score / longest
    found.sort(key=lambda f: (f.start, -(f.end - f.start)))
    out: list[Finding] = []
    for f in found:
        if out and f.start < out[-1].end:
            if f.score > out[-1].score:
                out[-1] = f
            continue
        out.append(f)
    return out


def redact(text: str, findings: list[Finding] | None = None) -> tuple[str, list[Finding]]:
    findings = detect_pii(text) if findings is None else findings
    out, last = [], 0
    for f in sorted(findings, key=lambda x: x.start):
        out.append(text[last:f.start])
        out.append(f"[{f.kind}]")
        last = f.end
    out.append(text[last:])
    return "".join(out), findings


# --------------------------------------------------------------- prompt injection
INJECTION_RULES: list[tuple[str, re.Pattern, float]] = [
    ("instruction-override", re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,30}\b(instruction|prompt|rule|directive|context)s?\b", re.I), 0.9),
    ("system-prompt-exfil", re.compile(r"\b(reveal|show|print|repeat|leak|output)\b.{0,30}\b(system prompt|hidden prompt|initial instructions|your instructions)\b", re.I), 0.9),
    ("role-hijack", re.compile(r"\b(you are now|act as|pretend (to be|you are)|from now on you)\b.{0,40}\b(dan|jailbroken|unrestricted|no restrictions|developer mode)\b", re.I), 0.85),
    ("jailbreak-keyword", re.compile(r"\b(DAN mode|developer mode|do anything now|jailbreak)\b", re.I), 0.7),
    ("delimiter-injection", re.compile(r"(<\|?(system|im_start|im_end)\|?>|\[/?INST\]|###\s*(system|instruction))", re.I), 0.8),
    ("exfil-url", re.compile(r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*(data|q|secret)=", re.I), 0.8),
    ("tool-abuse", re.compile(r"\b(send|email|post|upload)\b.{0,40}\b(to|at)\b.{0,30}(https?://|@)", re.I), 0.5),
    ("encoded-payload", re.compile(r"\b(base64|rot13|hex)\b.{0,20}\b(decode|decoded|follow)\b", re.I), 0.6),
]

BLOCKED_TOPICS: dict[str, re.Pattern] = {
    "weapons-synthesis": re.compile(r"\b(build|make|synthesi[sz]e)\b.{0,25}\b(bomb|explosive|nerve agent|bioweapon)\b", re.I),
    "self-harm-method": re.compile(r"\b(how to|ways to)\b.{0,15}\b(kill myself|commit suicide)\b", re.I),
    "malware": re.compile(r"\b(write|create|build)\b.{0,25}\b(ransomware|keylogger|malware)\b", re.I),
}


@dataclass
class Verdict:
    allowed: bool
    risk: float
    reasons: list[str]

    def to_dict(self):
        return asdict(self)


def scan_injection(text: str) -> Verdict:
    hits = [(n, s) for n, p, s in INJECTION_RULES if p.search(text)]
    risk = 0.0
    for _, s in hits:
        risk = 1 - (1 - risk) * (1 - s)  # noisy-OR combination
    return Verdict(allowed=risk < 0.6, risk=round(risk, 3), reasons=[n for n, _ in hits])


def content_filter(text: str) -> Verdict:
    hits = [n for n, p in BLOCKED_TOPICS.items() if p.search(text)]
    return Verdict(allowed=not hits, risk=1.0 if hits else 0.0, reasons=hits)


def check_input(query: str) -> dict:
    inj, cf = scan_injection(query), content_filter(query)
    pii = detect_pii(query)
    clean, _ = redact(query, pii)
    allowed = inj.allowed and cf.allowed
    return {
        "allowed": allowed,
        "injection": inj.to_dict(),
        "content": cf.to_dict(),
        "pii": [f.to_dict() for f in pii],
        "sanitized_query": clean,
        "block_reason": None if allowed else ("prompt-injection" if not inj.allowed else "blocked-content"),
    }


def sanitize_context(text: str) -> tuple[str, Verdict]:
    """Indirect-injection defense: neutralize instruction-like sentences inside retrieved docs."""
    verdict = scan_injection(text)
    if verdict.risk == 0:
        return text, verdict
    parts = re.split(r"(?<=[.!?\n])\s+", text)
    safe = [("[removed: suspicious instruction in source]" if not scan_injection(p).allowed else p) for p in parts]
    return " ".join(safe), verdict


def check_output(answer: str) -> dict:
    pii = detect_pii(answer)
    clean, _ = redact(answer, pii)
    leak = bool(re.search(r"system prompt|my instructions are", answer, re.I))
    return {"pii": [f.to_dict() for f in pii], "sanitized_answer": clean, "leak_suspected": leak}
