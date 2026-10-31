"""Analyzer engine construction: fail-closed spaCy backend selection.

Preference order for the NLP backend:

1. ``en_core_web_lg``  -- best NER quality for PERSON / LOCATION.
2. ``en_core_web_sm``  -- small model, still provides NER.

``PHI_SHIELD_SPACY_MODEL`` can name a preferred model; it is tried
first but never suppresses the preference list -- a bad pin used to
collapse the candidates to that single name and silently drop an
installed fallback (security-audit 2026-10-31, finding D1).

Fail-closed policy (finding D1): without a working NER model the
pipeline cannot see free-form PERSON / LOCATION mentions, so running
pattern-only would return narrative PHI unredacted under a success
response.  ``build_analyzer`` therefore raises ``RuntimeError`` when
no usable model is found.  The pattern-only engine (Presidio built-in
pattern recognizers plus the custom recognizers in
:mod:`phi_shield.recognizers`; title-anchored headers still caught,
free-form body mentions NOT caught) remains available only as an
explicit operator opt-in via ``PHI_SHIELD_ALLOW_PATTERN_ONLY=1``;
backends built that way are marked degraded (see
:func:`backend_is_degraded`) and the API surfaces that marker.

Model supply chain: the models are pinned as exact wheels with
sha256 hashes in requirements.txt (never ``spacy download``, which
fetches an unpinned, unhashed current release).  At selection time
each installed model is additionally checked against the spaCy
version range its own package metadata declares, so a model built
for a different spaCy line is rejected instead of loaded.
"""

import importlib.metadata
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

#: Backend strings for the pattern-only fallback start with this prefix;
#: it is the single marker the rest of the stack uses to detect the
#: degraded state.
DEGRADED_BACKEND_PREFIX = "pattern-only"

_ALLOW_PATTERN_ONLY_ENV = "PHI_SHIELD_ALLOW_PATTERN_ONLY"


def backend_is_degraded(backend):
    """True when *backend* describes the pattern-only fallback."""
    return bool(backend) and backend.startswith(DEGRADED_BACKEND_PREFIX)


def _allow_pattern_only():
    return os.environ.get(_ALLOW_PATTERN_ONLY_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _model_spacy_compatible(model_name):
    """True if the installed model package supports the running spaCy.

    The model's own package metadata declares a spaCy version range
    (e.g. ``spacy <3.8.0,>=3.7.2``); a model installed for a different
    spaCy line is a mis-pinned deployment and is rejected here rather
    than loaded.  Models that declare no spaCy constraint are left to
    the load attempt itself.
    """
    from packaging.requirements import Requirement

    try:
        md = importlib.metadata.metadata(model_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    running = importlib.metadata.version("spacy")
    for raw in md.get_all("Requires-Dist") or []:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.name.lower().replace("_", "-") == "spacy":
            return running in req.specifier
    return True


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
    """Create an AnalyzerEngine; returns (engine, backend_description).

    Raises RuntimeError when no usable spaCy NER model is available,
    unless the operator explicitly opted into degraded pattern-only
    operation with PHI_SHIELD_ALLOW_PATTERN_ONLY=1 (fail-closed,
    security-audit 2026-10-31 finding D1).
    """
    pinned = os.environ.get("PHI_SHIELD_SPACY_MODEL") or None
    candidates = ([pinned] if pinned else []) + [
        m for m in _MODEL_PREFERENCE if m != pinned
    ]
    backend = None
    nlp_engine = None
    import spacy.util

    for model in candidates:
        if not spacy.util.is_package(model):
            logger.warning("spaCy model %s not installed; skipping", model)
            continue
        if not _model_spacy_compatible(model):
            logger.warning(
                "spaCy model %s declares no support for the installed "
                "spaCy version; skipping",
                model,
            )
            continue
        try:
            nlp_engine = _spacy_engine(model)
            backend = f"spacy:{model}"
            logger.info(
                "spaCy model %s version %s",
                model,
                importlib.metadata.version(model),
            )
            break
        except Exception as exc:  # model present but failed to load
            logger.warning("spaCy model %s failed to load: %s", model, exc)
    if nlp_engine is None:
        if _allow_pattern_only():
            nlp_engine = _pattern_only_engine()
            backend = (
                "pattern-only DEGRADED (no spaCy NER model available; "
                "explicitly enabled via PHI_SHIELD_ALLOW_PATTERN_ONLY)"
            )
            logger.error(
                "PHI-Shield running DEGRADED pattern-only backend: "
                "free-text PERSON/LOCATION PHI may pass through "
                "unredacted. Responses are marked degraded."
            )
        else:
            raise RuntimeError(
                "No usable spaCy NER model available (tried: %s). "
                "Refusing to run pattern-only: free-text PERSON/LOCATION "
                "PHI would pass through unredacted under a success "
                "response. Install a pinned model (requirements.txt) or "
                "set PHI_SHIELD_ALLOW_PATTERN_ONLY=1 to explicitly "
                "accept degraded pattern-only operation."
                % (", ".join(candidates) or "none")
            )
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
