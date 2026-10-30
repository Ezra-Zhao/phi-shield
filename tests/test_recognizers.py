"""Custom recognizer coverage: MRN, account, device, license, ZIP, fax, age."""


def types_in(deid, text):
    return {e["entity_type"]: e["text"] for e in
            ({"entity_type": r.entity_type, "text": text[r.start:r.end]}
             for r in deid.detect(text))}


def test_mrn_detected(deid):
    found = types_in(deid, "MRN: 8473625 was verified at intake.")
    assert found.get("MEDICAL_RECORD_NUMBER") == "8473625"


def test_mrn_prefixed_value_detected(deid):
    found = types_in(deid, "Chart MRN: MRN-8473625 was verified at intake.")
    assert found.get("MEDICAL_RECORD_NUMBER") == "8473625"


def test_health_plan_account_detected(deid):
    found = types_in(deid, "Health Plan ID: HP-48291037 is active.")
    assert found.get("ACCOUNT_NUMBER") == "HP-48291037"


def test_account_number_detected(deid):
    found = types_in(deid, "Account No: 991827364 on file.")
    assert found.get("ACCOUNT_NUMBER") == "991827364"


def test_device_id_detected(deid):
    found = types_in(deid, "Device ID: PM-8841207 interrogated, battery OK.")
    assert found.get("DEVICE_ID") == "PM-8841207"


def test_device_serial_detected(deid):
    found = types_in(deid, "Serial No: SN-2024-88123 was recorded.")
    assert found.get("DEVICE_ID") == "SN-2024-88123"


def test_license_detected(deid):
    found = types_in(deid, "License #: A482910\nDEA: FM4829103")
    assert found.get("LICENSE_NUMBER") in ("A482910", "FM4829103")


def test_zip_detected(deid):
    found = types_in(deid, "San Jose, CA 95129")
    assert found.get("ZIP_CODE") == "95129"


def test_fax_not_confused_with_phone(deid):
    ents = deid.detect("Fax: (408) 555-0176")
    types = {r.entity_type for r in ents}
    assert "FAX_NUMBER" in types


def test_age_over_89_detected_younger_not(deid):
    old = deid.detect("The patient is a 92-year-old male.")
    assert any(r.entity_type == "AGE" for r in old)
    young = deid.detect("The patient is a 67-year-old male.")
    assert not any(r.entity_type == "AGE" for r in young)


def test_builtin_types(deid):
    text = ("Contact jane.doe@examplemail.com or (408) 555-0134. "
            "SSN: 123-45-6789. Portal: https://portal.example.org/x "
            "from 73.170.22.8 on 03/14/1958.")
    types = {r.entity_type for r in deid.detect(text)}
    for expected in ("EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "URL",
                     "IP_ADDRESS", "DATE_TIME"):
        assert expected in types, f"missing {expected} in {types}"
