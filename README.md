# PHI-Shield

A clinical-text PHI detection and de-identification pipeline in the style
of the **HIPAA Safe Harbor** method (45 CFR 164.514(b)(2)): it finds
protected health information in English clinical notes and removes or
generalizes it.

Built on **Microsoft Presidio** (`presidio-analyzer` + `presidio-anonymizer`)
with spaCy NER, plus custom pattern recognizers for the clinical
identifiers Presidio does not ship (MRN, health-plan / account numbers,
device IDs, license numbers, VINs, ZIP codes, fax numbers, street
addresses, ages over 89) and title-anchored name patterns that back up NER.

> **Synthetic data only.** Every note in the evaluation set is generated
> by `scripts/generate_dataset.py` from invented names, 555-prefix phone
> numbers, example domains, and format-valid but fictitious ID numbers.
> No real patient data appears anywhere in this project.

## Output modes

| Mode | Behavior |
|---|---|
| `mask` | Every detected identifier becomes `***`. |
| `label` | Every detected identifier becomes a type label: `[PERSON]`, `[DATE]`, `[MRN]`, `[SSN]`, ... |
| `safe_harbor` | Safe Harbor transform: dates -> **year only**; ZIP codes -> **first 3 digits** (rest zeroed; `000`-prefix areas -> `00000`); ages **over 89 -> 90**; all other identifiers are removed (type label left in place). |

## HIPAA Safe Harbor: the 18 identifier types

| # | Identifier (45 CFR 164.514(b)(2)(i)) | How PHI-Shield handles it |
|---|---|---|
| 1 | Names | spaCy NER `PERSON` + title-anchored patterns (`Patient:`, `Dr.`, `Mr./Mrs./Ms.`, `Attending:`) |
| 2 | Geographic subdivisions smaller than state (street, city, ZIP) | spaCy NER `LOCATION` (city) + custom street-address pattern + custom `ZIP_CODE` recognizer; Safe Harbor mode truncates ZIP to 3 digits |
| 3 | Dates (except year) related to an individual | Presidio `DATE_TIME`; Safe Harbor mode reduces to year only |
| 4 | Phone numbers | Presidio `PHONE_NUMBER` |
| 5 | Fax numbers | Custom `FAX_NUMBER` recognizer (label-anchored patterns) |
| 6 | Email addresses | Presidio `EMAIL_ADDRESS` |
| 7 | Social Security numbers | Presidio `US_SSN` (validates area/group/serial rules) |
| 8 | Medical record numbers | Custom `MEDICAL_RECORD_NUMBER` recognizer (`MRN:`-anchored) |
| 9 | Health plan beneficiary numbers | Custom `ACCOUNT_NUMBER` recognizer (`Health Plan ID:` / `Member ID:`) |
| 10 | Account numbers | Custom `ACCOUNT_NUMBER` recognizer (`Account No:` / `Account:`) |
| 11 | Certificate / license numbers | Custom `LICENSE_NUMBER` recognizer (`License #:` and DEA formats) |
| 12 | Vehicle identifiers and serial numbers (VIN) | Custom `VEHICLE_ID` recognizer (17-character VIN) |
| 13 | Device identifiers and serial numbers | Custom `DEVICE_ID` recognizer (`Device ID:`, `Serial No:` formats) |
| 14 | Web URLs | Presidio `URL` |
| 15 | IP addresses | Presidio `IP_ADDRESS` |
| 16 | Biometric identifiers | **Not covered** -- not present in text notes; requires a biometric system, out of scope for a text pipeline |
| 17 | Full-face photographs | **Not covered** -- image modality, out of scope for a text pipeline |
| 18 | Any other unique identifying number | Partially covered: ages over 89 (custom `AGE` recognizer, collapsed to 90 in Safe Harbor mode) plus the catch-all label removal of every detected type above; arbitrary unstructured unique numbers with no label or known format are **not** guaranteed to be caught |

Additional Safe Harbor rules implemented: ages over 89 are aggregated to
90 (identifier-adjacent rule in 164.514(b)(2)(i)(C)); ZIP codes whose
first three digits are `000` collapse to `00000`.

## Architecture

```
phi_shield/
  recognizers.py   custom PatternRecognizers (MRN, account, device,
                   license, VIN, ZIP, fax, street address, age>89,
                   title-anchored names)
  engine.py        AnalyzerEngine assembly + overlap resolution;
                   spaCy chain: en_core_web_lg -> en_core_web_sm;
                   fail-closed if no usable model (pattern-only only
                   via explicit PHI_SHIELD_ALLOW_PATTERN_ONLY=1 opt-in,
                   responses then marked degraded)
  deidentify.py    Deidentifier: mask / label via presidio-anonymizer,
                   safe_harbor via per-instance span transforms
scripts/
  generate_dataset.py   synthetic 200-note gold set -> data/synthetic_notes.jsonl
evaluate.py        entity-level P/R/F1 per type + overall -> eval_report.md
app.py             FastAPI service (POST /deidentify, GET /health, demo UI)
static/demo.html   one-page demo (paste text -> before/after)
tests/             pytest suite
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # includes BOTH spaCy models,
                                            # pinned wheels + sha256 hashes
```

The spaCy models are installed from requirements.txt as exact,
hash-pinned wheels -- do not substitute `spacy download` (unpinned,
unhashed). If no usable spaCy model is available at runtime, the
service **fails closed**: it refuses to start rather than silently
degrading to the pattern-only engine, which cannot see free-form
names/places and would return them unredacted under a success
response. Pattern-only operation remains available only as an
explicit opt-in (`PHI_SHIELD_ALLOW_PATTERN_ONLY=1`); every response
is then marked `"degraded": true` with a warning, and `GET /health`
reports `"status": "degraded"`. The active backend is always
reported by `GET /health` and printed in `eval_report.md`.

Request bounds: `POST /deidentify` accepts at most 100,000
characters of text (HTTP 413 beyond that, and for request bodies
over 1 MiB) and is rate limited per client (HTTP 429; default
60/minute, `PHI_SHIELD_RATE_LIMIT_PER_MINUTE`).

## Run

```bash
# (re)generate the synthetic evaluation set
.venv/bin/python scripts/generate_dataset.py            # 200 notes, seed 42

# evaluate (writes eval_report.md)
.venv/bin/python evaluate.py

# tests
.venv/bin/python -m pytest

# API + demo UI
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000/ for the demo,
# POST http://127.0.0.1:8000/deidentify  {"text": "...", "mode": "safe_harbor"}
```

## Evaluation results

<!-- EVAL_RESULTS -->
Evaluated 2026-10-30 on the bundled synthetic set (200 notes, 2,373
gold entities; NER backend: spaCy `en_core_web_sm`). Matching rule:
same type + prediction covers ≥50% of the gold span.

**Overall precision 0.945 · Overall recall 0.997 · Overall F1 0.970**
(headline target — overall recall ≥ 0.95 — met.)

| Type | P | R | F1 |
|---|---|---|---|
| PERSON | 0.897 | 0.998 | 0.945 |
| DATE_TIME | 0.873 | 0.997 | 0.931 |
| LOCATION | 0.979 | 0.986 | 0.983 |
| PHONE_NUMBER | 0.990 | 1.000 | 0.995 |
| MEDICAL_RECORD_NUMBER / ACCOUNT_NUMBER / DEVICE_ID / LICENSE_NUMBER / FAX_NUMBER / ZIP_CODE / AGE / US_SSN / EMAIL_ADDRESS / IP_ADDRESS / URL / VEHICLE_ID | 1.000 | 1.000 | 1.000 |

Full per-type counts: `eval_report.md`. Remaining errors are
precision-side: the small spaCy model over-tags some clinical terms as
PERSON, and Presidio flags some non-gold date-like strings. Only 6 of
2,373 gold entities were missed (4 LOCATION, 1 PERSON, 1 DATE_TIME).
<!-- /EVAL_RESULTS -->

Evaluation method: entity-level scoring against the generator's planted
gold spans. A prediction counts as a true positive when the entity type
matches and the predicted span covers at least 50% of the gold span
(greedy one-to-one matching).

## Limitations

- **Synthetic-data evaluation only.** The 200 notes are templated and far
  cleaner than real clinical text; scores here do not transfer directly
  to production EHR data.
- **Not a compliance certification.** Safe Harbor de-identification under
  HIPAA also requires that the covered entity have no actual knowledge
  that residual information can identify an individual; this tool is a
  proof-of-skill demo, not a certified de-identification solution.
- NER quality depends on the spaCy model in use (`lg` > `sm`). The
  pattern-only engine exists only as an explicitly opted-in degraded
  mode (see Setup); it is never entered silently. Handwriting, OCR
  noise, misspellings, and non-English text are out of scope.
- Ages are only flagged when over 89 and expressed in the covered surface
  forms (`92-year-old`, `age 92`, `aged 92`, `92 years old`).
