"""Custom Presidio pattern recognizers for clinical identifiers.

Presidio ships strong built-ins for PERSON / LOCATION (spaCy NER), DATE_TIME,
PHONE_NUMBER, EMAIL_ADDRESS, US_SSN, URL and IP_ADDRESS.  The recognizers
below cover the HIPAA Safe Harbor identifier classes that Presidio does not
ship out of the box, plus title-anchored name patterns that act as a
documented fallback / recall booster for spaCy NER.

All patterns are written against the surface forms used in clinical headers
("MRN: ...", "Fax: ...", "License #: ...").  Where a label precedes the
value, a fixed-width lookbehind keeps the matched span on the value itself.
"""

from presidio_analyzer import Pattern, PatternRecognizer


def _rec(entity, name, patterns, context=None):
    return PatternRecognizer(
        supported_entity=entity,
        name=name,
        patterns=patterns,
        context=context or [],
    )


def build_custom_recognizers():
    """Return the list of custom PatternRecognizer objects."""
    recognizers = []

    # --- 8. Medical record numbers -------------------------------------
    recognizers.append(
        _rec(
            "MEDICAL_RECORD_NUMBER",
            "Medical record number recognizer",
            [
                Pattern("mrn colon space", r"(?<=MRN: )\d{6,10}", 0.85),
                Pattern("mrn colon", r"(?<=MRN:)\d{6,10}", 0.85),
                Pattern("mrn hash", r"(?<=MRN #)\d{6,10}", 0.85),
                Pattern("mrn long label", r"(?<=Medical Record Number: )\d{6,10}", 0.85),
                Pattern("mrn space", r"(?<=MRN )\d{6,10}", 0.55),
                # value itself carries the prefix: "MRN: MRN-48291047"
                Pattern("mrn prefixed value", r"(?<=MRN-)\d{6,10}", 0.85),
            ],
            context=["mrn", "medical record", "chart number"],
        )
    )

    # --- 9/10. Health-plan beneficiary & account numbers ----------------
    recognizers.append(
        _rec(
            "ACCOUNT_NUMBER",
            "Health plan / account number recognizer",
            [
                Pattern("health plan id", r"(?<=Health Plan ID: )[A-Z]{2}-\d{8}", 0.85),
                Pattern("member id", r"(?<=Member ID: )[A-Z]{2}-\d{8}", 0.85),
                Pattern("account no", r"(?<=Account No: )\d{9}", 0.85),
                Pattern("account hash", r"(?<=Account # )\d{9}", 0.85),
                Pattern("account colon", r"(?<=Account: )\d{9}", 0.6),
            ],
            context=["account", "health plan", "member id", "beneficiary"],
        )
    )

    # --- 13. Device identifiers and serial numbers ----------------------
    recognizers.append(
        _rec(
            "DEVICE_ID",
            "Device identifier recognizer",
            [
                Pattern("device id label", r"(?<=Device ID: )[A-Z]{2}-\d{7}", 0.85),
                Pattern("device serial label", r"(?<=Device Serial: )[A-Z]{2}-\d{7}", 0.85),
                Pattern("serial no token", r"SN-\d{4}-\d{5}", 0.7),
                Pattern("implant token", r"(?<=Serial No: )SN-\d{4}-\d{5}", 0.85),
            ],
            context=["device", "serial", "implant", "pacemaker", "pump"],
        )
    )

    # --- 11. Certificate / license numbers ------------------------------
    recognizers.append(
        _rec(
            "LICENSE_NUMBER",
            "License / DEA number recognizer",
            [
                Pattern("license label", r"(?<=License #: )[A-Z]\d{6}", 0.85),
                Pattern("license colon", r"(?<=License: )[A-Z]\d{6}", 0.7),
                Pattern("dea colon", r"(?<=DEA: )[A-Z]{2}\d{7}", 0.85),
                Pattern("dea hash", r"(?<=DEA # )[A-Z]{2}\d{7}", 0.85),
            ],
            context=["license", "dea"],
        )
    )

    # --- 12. Vehicle identifiers (VIN) ----------------------------------
    recognizers.append(
        _rec(
            "VEHICLE_ID",
            "VIN recognizer",
            [
                Pattern("vin label", r"(?<=VIN: )[A-HJ-NPR-Z0-9]{17}", 0.85),
                Pattern("vin bare", r"\b[A-HJ-NPR-Z0-9]{17}\b", 0.4),
            ],
            context=["vin", "vehicle"],
        )
    )

    # --- 2. ZIP codes -----------------------------------------------------
    recognizers.append(
        _rec(
            "ZIP_CODE",
            "US ZIP code recognizer",
            [
                Pattern("zip plus4", r"\b\d{5}-\d{4}\b", 0.75),
                Pattern("zip label", r"(?<=ZIP: )\d{5}", 0.85),
                Pattern("zip bare", r"\b\d{5}\b", 0.45),
            ],
            context=["zip", "postal"],
        )
    )

    # --- 2. Street addresses (feeds the LOCATION entity) -----------------
    recognizers.append(
        _rec(
            "LOCATION",
            "Street address recognizer",
            [
                Pattern(
                    "street address",
                    r"\b\d{1,6}\s+(?:[A-Z][A-Za-z]+\s+){1,3}"
                    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Court|Ct|Way|Place|Pl)\b",
                    0.65,
                ),
            ],
            context=["street", "address", "ave", "blvd", "road"],
        )
    )

    # --- 5. Fax numbers ----------------------------------------------------
    recognizers.append(
        _rec(
            "FAX_NUMBER",
            "Fax number recognizer",
            [
                Pattern("fax paren", r"(?<=Fax: )\(\d{3}\) \d{3}-\d{4}", 0.9),
                Pattern("fax dash", r"(?<=Fax: )\d{3}-\d{3}-\d{4}", 0.9),
                Pattern("fax dot", r"(?<=Fax: )\d{3}\.\d{3}\.\d{4}", 0.9),
                Pattern("fax plain paren", r"(?<=Fax )\(\d{3}\) \d{3}-\d{4}", 0.8),
            ],
            context=["fax", "facsimile"],
        )
    )

    # --- Ages over 89 (Safe Harbor identifier #18-adjacent / age rule) -----
    # Only the digits are matched; ages <= 89 are not restricted by Safe
    # Harbor and are intentionally not flagged.
    recognizers.append(
        _rec(
            "AGE",
            "Age over 89 recognizer",
            [
                Pattern("age word", r"(?<=\bage )(9\d|10\d)\b", 0.8),
                Pattern("aged word", r"(?<=\baged )(9\d|10\d)\b", 0.8),
                Pattern("year old", r"\b(9\d|10\d)(?=-year-old)", 0.8),
                Pattern("years old", r"\b(9\d|10\d)(?= years old)", 0.8),
            ],
            context=["age", "aged", "year-old", "years old"],
        )
    )

    # --- SSN: format-based supplement --------------------------------------
    # Presidio's built-in US_SSN recognizer applies validity heuristics
    # that reject some SSN-shaped strings (e.g. the textbook example
    # 123-45-6789). For Safe Harbor purposes any ddd-dd-dddd string is
    # treated as an SSN candidate; duplicates collapse downstream.
    recognizers.append(
        _rec(
            "US_SSN",
            "SSN format recognizer",
            [Pattern("ssn dashed", r"\b\d{3}-\d{2}-\d{4}\b", 0.5)],
            context=["ssn", "social security", "social"],
        )
    )

    # --- Dates: month-name supplement -------------------------------------
    # Presidio's built-in DATE_TIME recognizer intermittently misses
    # abbreviated month-name dates ("Apr 27, 2026") depending on context.
    # These deterministic patterns close that gap; duplicates with the
    # built-in recognizer collapse during overlap resolution.
    _months_full = (
        "January|February|March|April|May|June|July|August|September|"
        "October|November|December"
    )
    _months_abbr = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    recognizers.append(
        _rec(
            "DATE_TIME",
            "Month-name date recognizer",
            [
                Pattern("month abbr date",
                        rf"\b(?:{_months_abbr})\.? \d{{1,2}}, \d{{4}}\b", 0.6),
                Pattern("month full date",
                        rf"\b(?:{_months_full}) \d{{1,2}}, \d{{4}}\b", 0.6),
            ],
            context=["date", "dob", "admission", "discharge", "visit"],
        )
    )

    # --- Names: title-anchored fallback / recall booster for spaCy NER -----
    # These fire on the strongly-cued name slots used in clinical documents
    # (headers, salutations, signatures).  spaCy NER covers free-form
    # mentions; these patterns cover the templated ones even when the NER
    # model is small -- or absent (pattern-only fallback engine).
    recognizers.append(
        _rec(
            "PERSON",
            "Title-anchored name recognizer",
            [
                Pattern("patient header", r"(?<=Patient: )[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+", 0.85),
                Pattern("patient name header", r"(?<=Patient Name: )[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+", 0.85),
                Pattern("doctor", r"(?<=Dr\. )[A-Z][a-z]+ [A-Z][a-z]+", 0.85),
                Pattern("attending", r"(?<=Attending: )[A-Z][a-z]+ [A-Z][a-z]+", 0.8),
                Pattern("mr surname", r"(?<=Mr\. )[A-Z][a-z]+", 0.8),
                Pattern("mrs surname", r"(?<=Mrs\. )[A-Z][a-z]+", 0.8),
                Pattern("ms surname", r"(?<=Ms\. )[A-Z][a-z]+", 0.8),
                Pattern("dear name", r"(?<=Dear )[A-Z][a-z]+ [A-Z][a-z]+", 0.7),
            ],
            context=["patient", "dr", "mr", "mrs", "ms", "attending"],
        )
    )

    return recognizers
