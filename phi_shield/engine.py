"""Analyzer engine construction with a documented spaCy fallback chain.

Preference order for the NLP backend:

1. ``en_core_web_lg``  -- best NER quality for PERSON / LOCATION.
2. ``en_core_web_sm``  -- small model, still provides NER.
3. Pattern-only stub   -- a minimal ``NlpEngine`` that returns no NER
   entities.  The pipeline still runs end-to-end on Presidio's built-in
   pattern recognizers (dates, phones, emails, SSNs, URLs, IPs) plus the
   custom recognizers in :mod:`phi_shield.recognizers` (which include
   title-anchored name patterns, so headers/salutations are still caught).
   Free-form person/location mentions in body text are the known casualty
   of this path.

``PHI_SHIELD_SPACY_MODEL`` can pin a specific model name.
"""

import logging
import os
import re

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngine, NlpEngineProvider

from .recognizers import build_custom_recognizers

logger = logging.getLogger(__name__)

SUPPORTED_ENTITIES = [
    "PERSON",
    "LOCATION",
    "DATE_TIME",
    "PHONE_NUMBER",
    "FAX_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "MEDICAL_RECORD_NUMBER",
    "ACCOUNT_NUMBER",
    "LICENSE_NUMBER",
    "VEHICLE_ID",
    "DEVICE_ID",
    "ZIP_CODE",
    "URL",
    "IP_ADDRESS",
    "AGE",
]

_MODEL_PREFERENCE = ["en_core_web_lg", "en_core_web_sm"]


class _PatternOnlyArtifacts:
    """Duck-typed stand-in is not enough for Presidio; see engine below."""


def _pattern_only_engine():
    """Build an AnalyzerEngine backed by a no-NER NLP stub.

    Uses ``spacy.blank("en")`` (tokenizer only, no trained pipeline) so
    Presidio still gets real token artifacts and context enhancement,
    while NER contributes nothing.  Pattern recognizers -- Presidio's
    built-ins and the custom ones in this package -- carry the pipeline.
    """
    import spacy
    from presidio_analyzer.nlp_engine import NlpArtifacts

    blank = spacy.blank("en")

    class PatternOnlyNlpEngine(NlpEngine):
        def load(self):
            self._loaded = True

        def is_loaded(self):
            return getattr(self, "_loaded", False)

        def is_stopword(self, word, language):
            return word.lower() in blank.Defaults.stop_words

        def is_punct(self, word, language):
            return all(not c.isalnum() for c in word)

        def get_supported_entities(self):
            return []

        def get_supported_languages(self):
            return ["en"]

        def process_text(self, text, language):
            doc = blank(text)
            return NlpArtifacts(
                entities=[],
                tokens=doc,
                tokens_indices=[tok.idx for tok in doc],
                lemmas=[tok.text for tok in doc],
                nlp_engine=self,
                language=language,
            )

        def process_batch(self, texts, language, as_tuples=False):
            for text in texts:
                artifacts = self.process_text(text, language)
                yield (text, artifacts) if as_tuples else artifacts

    engine = PatternOnlyNlpEngine()
    engine.load()
    return engine


def _spacy_engine(model_name):
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model_name}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    return provider.create_engine()


def build_analyzer():
    """Create an AnalyzerEngine; returns (engine, backend_description)."""
    pinned = os.environ.get("PHI_SHIELD_SPACY_MODEL")
    candidates = [pinned] if pinned else list(_MODEL_PREFERENCE)
    backend = None
    nlp_engine = None
    import spacy.util

    for model in candidates:
        if not model:
            continue
        if not spacy.util.is_package(model):
            logger.warning("spaCy model %s not installed; skipping", model)
            continue
        try:
            nlp_engine = _spacy_engine(model)
            backend = f"spacy:{model}"
            break
        except Exception as exc:  # model present but failed to load
            logger.warning("spaCy model %s failed to load: %s", model, exc)
    if nlp_engine is None:
        nlp_engine = _pattern_only_engine()
        backend = "pattern-only (no spaCy model available)"
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    for recognizer in build_custom_recognizers():
        analyzer.registry.add_recognizer(recognizer)
    logger.info("PHI-Shield analyzer backend: %s", backend)
    return analyzer, backend


# Presidio's DATE_TIME recognizer also fires on durations and fragments
# ("24 hours", "3 months", "last month", "90-year-old", bare 5-digit ZIPs,
# phone suffixes).  Those are not calendar dates, so DATE_TIME results are
# kept only when the span actually looks like a calendar date: a numeric
# date, an ISO date, a month name, or a bare 4-digit year (a year alone is
# a legitimate Safe Harbor date element).
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_PLAUSIBLE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{1,2}-\d{1,2}|\b(?:19|20)\d{2}\b|\b(?:"
    + _MONTHS
    + r")\b)",
    re.IGNORECASE,
)

_LOCATION_STOPLIST = {"MD", "DO", "RN", "IV", "PO", "IM", "ED", "BP", "HR"}

# When spans collide, identifier-specific recognizers (anchored patterns
# for MRN / account / device / license / ZIP / fax / age) are more reliable
# than broad NER or date spans that happen to swallow the same characters
# (e.g. NER tagging "CA 95129-4417" as a PERSON, or the date recognizer
# claiming a bare ZIP code).  Higher = wins the overlap.
_SPECIFICITY = {
    "FAX_NUMBER": 5,
    "MEDICAL_RECORD_NUMBER": 4,
    "ACCOUNT_NUMBER": 4,
    "DEVICE_ID": 4,
    "LICENSE_NUMBER": 4,
    "VEHICLE_ID": 4,
    "ZIP_CODE": 4,
    "AGE": 4,
    "US_SSN": 3,
    "EMAIL_ADDRESS": 3,
    "URL": 3,
    "IP_ADDRESS": 3,
    "PHONE_NUMBER": 3,
    "DATE_TIME": 2,
    "PERSON": 1,
    "LOCATION": 1,
}


def analyze(analyzer, text):
    """Analyze text and resolve overlapping results deterministically.

    Steps: (1) drop implausible DATE_TIME spans (durations/fragments, see
    above); (2) greedy overlap resolution ordered by recognizer
    specificity, then score, then span length.
    """
    raw = analyzer.analyze(
        text=text,
        language="en",
        entities=SUPPORTED_ENTITIES,
        return_decision_process=False,
    )
    def _plausible(res):
        span = text[res.start : res.end]
        if res.entity_type == "DATE_TIME":
            # a single calendar date never spans lines; the built-in
            # recognizer occasionally merges two dates across a newline
            return "\n" not in span and bool(_DATE_PLAUSIBLE.search(span))
        if res.entity_type in ("PERSON", "LOCATION"):
            # NER run-ons across line breaks are artifacts, and the small
            # spaCy model systematically tags clinical abbreviations
            # ("MD", "IV") and lowercase body words as locations.
            if "\n" in span:
                return False
            if res.entity_type == "LOCATION":
                token = span.strip()
                if token in _LOCATION_STOPLIST:
                    return False
                if token.isalpha() and token.islower():
                    return False
        return True

    filtered = [r for r in raw if _plausible(r)]
    ordered = sorted(
        filtered,
        key=lambda r: (
            -_SPECIFICITY.get(r.entity_type, 0),
            -r.score,
            -(r.end - r.start),
        ),
    )
    kept = []
    for res in ordered:
        if any(not (res.end <= k.start or res.start >= k.end) for k in kept):
            continue
        kept.append(res)
    return sorted(kept, key=lambda r: r.start)
