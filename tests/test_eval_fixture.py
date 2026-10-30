"""Small gold fixture through the real evaluation scorer."""

from evaluate import prf, score_records
from phi_shield.engine import build_analyzer


def make_record(text, spans):
    entities = []
    for value, etype in spans:
        start = text.index(value)
        entities.append({"start": start, "end": start + len(value),
                         "entity_type": etype, "text": value})
    return {"id": "fix", "note_type": "fixture", "text": text, "entities": entities}


FIXTURES = [
    make_record(
        "Patient: John Smith\nDOB: 03/14/1958\nMRN: 8473625\nPhone: (408) 555-0134",
        [("John Smith", "PERSON"), ("03/14/1958", "DATE_TIME"),
         ("8473625", "MEDICAL_RECORD_NUMBER"), ("(408) 555-0134", "PHONE_NUMBER")],
    ),
    make_record(
        "Fax: 408-555-0176\nEmail: jane.doe@examplemail.com\nSan Jose, CA 95129",
        [("408-555-0176", "FAX_NUMBER"), ("jane.doe@examplemail.com", "EMAIL_ADDRESS"),
         ("San Jose", "LOCATION"), ("95129", "ZIP_CODE")],
    ),
    make_record(
        "The patient is a 93-year-old. Device ID: PM-8841207. SSN: 123-45-6789.",
        [("93", "AGE"), ("PM-8841207", "DEVICE_ID"), ("123-45-6789", "US_SSN")],
    ),
]


def test_fixture_recall_is_perfect():
    analyzer, _backend = build_analyzer()
    counts = score_records(FIXTURES, analyzer)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for c in counts.values():
        for k in totals:
            totals[k] += c[k]
    _p, recall, _f1 = prf(totals)
    assert totals["fn"] == 0, f"missed entities: { {k: v for k, v in counts.items() if v['fn']} }"
    assert recall == 1.0
