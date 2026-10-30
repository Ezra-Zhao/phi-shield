"""De-identification: three output modes over Presidio analyzer results.

Modes
-----
mask
    Every detected identifier is replaced with ``***``
    (via presidio-anonymizer's ``replace`` operator).
label
    Every detected identifier is replaced with a bracketed type label,
    e.g. ``[PERSON]``, ``[DATE]``, ``[MRN]`` (presidio-anonymizer).
safe_harbor
    HIPAA Safe Harbor style transform:
      * dates            -> year only (``03/14/1958`` -> ``1958``)
      * ZIP codes        -> first 3 digits, remainder zeroed (``95129`` ->
        ``95100``; ZIP areas whose first three digits are ``000`` collapse
        to ``00000``, per 45 CFR 164.514(b)(2)(i)(B))
      * ages over 89     -> ``90``
      * everything else  -> bracketed type label (value removed)

    Safe Harbor needs per-instance replacement values (each date has its
    own year), which presidio-anonymizer's static per-type operators
    cannot express, so this mode applies the transforms directly over the
    analyzer spans.  Mask and label modes go through
    :class:`presidio_anonymizer.AnonymizerEngine`.
"""

import re

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .engine import analyze, build_analyzer

LABEL_MAP = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "DATE_TIME": "DATE",
    "PHONE_NUMBER": "PHONE",
    "FAX_NUMBER": "FAX",
    "EMAIL_ADDRESS": "EMAIL",
    "US_SSN": "SSN",
    "MEDICAL_RECORD_NUMBER": "MRN",
    "ACCOUNT_NUMBER": "ACCOUNT",
    "LICENSE_NUMBER": "LICENSE",
    "VEHICLE_ID": "VIN",
    "DEVICE_ID": "DEVICE",
    "ZIP_CODE": "ZIP",
    "URL": "URL",
    "IP_ADDRESS": "IP",
    "AGE": "AGE",
}

MODES = ("mask", "label", "safe_harbor")

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def label_for(entity_type):
    return "[%s]" % LABEL_MAP.get(entity_type, entity_type)


def safe_harbor_value(entity_type, original_text):
    """Replacement string for one span under the Safe Harbor transform."""
    if entity_type == "DATE_TIME":
        match = _YEAR_RE.search(original_text)
        return match.group(0) if match else label_for(entity_type)
    if entity_type == "ZIP_CODE":
        digits = re.sub(r"\D", "", original_text)
        if len(digits) >= 5:
            head = digits[:3]
            return "00000" if head == "000" else head + "00"
        return label_for(entity_type)
    if entity_type == "AGE":
        return "90"
    return label_for(entity_type)


class Deidentifier:
    """Stateful wrapper: builds the analyzer once, reuses it per call."""

    def __init__(self, analyzer=None):
        if analyzer is None:
            analyzer, backend = build_analyzer()
        else:
            backend = "external"
        self.analyzer = analyzer
        self.backend = backend
        self._anonymizer = AnonymizerEngine()

    def detect(self, text):
        return analyze(self.analyzer, text)

    def deidentify(self, text, mode="label"):
        """Return (deidentified_text, entities) for the given mode."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        results = self.detect(text)
        if mode == "mask":
            operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "***"})}
            out = self._anonymizer.anonymize(
                text=text, analyzer_results=results, operators=operators
            ).text
        elif mode == "label":
            operators = {
                etype: OperatorConfig("replace", {"new_value": label_for(etype)})
                for etype in {r.entity_type for r in results}
            }
            operators["DEFAULT"] = OperatorConfig(
                "replace", {"new_value": "[PHI]"}
            )
            out = self._anonymizer.anonymize(
                text=text, analyzer_results=results, operators=operators
            ).text
        else:  # safe_harbor: per-instance transforms over spans
            out = text
            for r in sorted(results, key=lambda x: x.start, reverse=True):
                replacement = safe_harbor_value(
                    r.entity_type, text[r.start : r.end]
                )
                out = out[: r.start] + replacement + out[r.end :]
        entities = [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(float(r.score), 3),
                "text": text[r.start : r.end],
            }
            for r in results
        ]
        return out, entities
