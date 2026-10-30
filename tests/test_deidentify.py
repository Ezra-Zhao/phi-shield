"""Output-mode behavior: mask, label, and the Safe Harbor transforms."""

from phi_shield.deidentify import safe_harbor_value

SAMPLE = (
    "Patient: John Smith\n"
    "DOB: 03/14/1958\n"
    "MRN: 8473625\n"
    "Address: 482 Maple Avenue, San Jose, CA 95129\n"
    "The patient is a 92-year-old male."
)


def test_mask_mode(deid):
    out, entities = deid.deidentify(SAMPLE, mode="mask")
    assert "John Smith" not in out
    assert "8473625" not in out
    assert "03/14/1958" not in out
    assert "***" in out
    assert entities, "expected detected entities to be returned"


def test_label_mode(deid):
    out, _ = deid.deidentify(SAMPLE, mode="label")
    assert "[PERSON]" in out
    assert "[MRN]" in out
    assert "[DATE]" in out
    assert "John Smith" not in out


def test_safe_harbor_mode_end_to_end(deid):
    out, _ = deid.deidentify(SAMPLE, mode="safe_harbor")
    assert "1958" in out and "03/14" not in out          # date -> year only
    assert "95100" in out and "95129" not in out          # ZIP -> first 3 + 00
    assert "90-year-old" in out and "92-year-old" not in out  # age > 89 -> 90
    assert "[MRN]" in out                                  # other IDs removed


def test_safe_harbor_value_units():
    assert safe_harbor_value("DATE_TIME", "March 14, 1958") == "1958"
    assert safe_harbor_value("DATE_TIME", "2026-03-06") == "2026"
    assert safe_harbor_value("ZIP_CODE", "95129") == "95100"
    assert safe_harbor_value("ZIP_CODE", "95129-4417") == "95100"
    assert safe_harbor_value("ZIP_CODE", "00012") == "00000"  # 000-prefix rule
    assert safe_harbor_value("AGE", "93") == "90"
    assert safe_harbor_value("PERSON", "John Smith") == "[PERSON]"


def test_unknown_mode_rejected(deid):
    try:
        deid.deidentify(SAMPLE, mode="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown mode")
