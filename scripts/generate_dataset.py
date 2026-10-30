#!/usr/bin/env python3
"""Generate the synthetic PHI evaluation set.

Creates 200 short English clinical notes (discharge summaries, referral
letters, clinic progress notes, telehealth notes) with planted PHI spans
and exact ground-truth labels, written as JSONL:

    {"id": ..., "note_type": ..., "text": ...,
     "entities": [{"start": int, "end": int, "entity_type": str, "text": str}]}

ALL data is synthetic: invented names, 555-prefix phone numbers, example
domains, and format-valid but fictitious ID numbers.  No real patient
data is used anywhere.

Run:  python scripts/generate_dataset.py [--count 200] [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
    "Nancy", "Matthew", "Lisa", "Anthony", "Betty", "Mark", "Margaret",
    "Donald", "Sandra", "Steven", "Ashley", "Paul", "Kimberly", "Andrew",
    "Emily", "Kevin", "Michelle", "Brian", "Dorothy", "Timothy", "Amanda",
    "Priya", "Rajesh", "Wei", "Mei", "Hiroshi", "Yuki", "Carlos", "Ana",
    "Fatima", "Omar", "Olga", "Ivan", "Ingrid", "Lars",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Patel", "Chen", "Kim", "Tanaka", "Haddad", "Novak",
]
TITLES = ["Mr.", "Mrs.", "Ms."]
CITIES = [
    ("San Jose", "CA", "951"), ("Sacramento", "CA", "958"),
    ("Austin", "TX", "787"), ("Denver", "CO", "802"),
    ("Miami", "FL", "331"), ("Chicago", "IL", "606"),
    ("Seattle", "WA", "981"), ("Portland", "OR", "972"),
    ("Phoenix", "AZ", "850"), ("Boston", "MA", "021"),
    ("Fresno", "CA", "937"), ("Tampa", "FL", "336"),
]
STREETS = [
    "Maple Avenue", "Oak Street", "Cedar Lane", "Washington Boulevard",
    "Elm Drive", "Lincoln Road", "Park Avenue", "Hill Court",
    "River Road", "Sunset Drive", "Walnut Street", "Birch Lane",
]
AREA_CODES = ["408", "512", "303", "206", "617", "305", "773", "503", "602", "916"]
CONDITIONS = [
    "type 2 diabetes mellitus", "hypertension", "community-acquired pneumonia",
    "congestive heart failure", "chronic obstructive pulmonary disease",
    "atrial fibrillation", "a fractured left hip", "appendicitis",
    "chronic kidney disease", "coronary artery disease",
]
PROCEDURES = [
    "underwent a laparoscopic appendectomy", "received IV antibiotics",
    "was started on oral metformin", "underwent cardiac catheterization",
    "received physical therapy twice daily", "was placed on telemetry",
    "underwent a total hip replacement", "received a blood transfusion",
]


class NoteBuilder:
    """Accumulates text while recording exact PHI span offsets."""

    def __init__(self):
        self.parts = []
        self.entities = []
        self._len = 0

    def add(self, text):
        self.parts.append(text)
        self._len += len(text)
        return self

    def phi(self, value, entity_type):
        start = self._len
        self.add(value)
        self.entities.append(
            {"start": start, "end": start + len(value),
             "entity_type": entity_type, "text": value}
        )
        return self

    def build(self):
        return "".join(self.parts)


class Person:
    def __init__(self, rng):
        self.first = rng.choice(FIRST_NAMES)
        self.last = rng.choice(LAST_NAMES)
        self.title = rng.choice(TITLES)
        self.middle = rng.choice("ABCDEFGHJKLMNPRSTW") if rng.random() < 0.3 else None

    @property
    def full(self):
        if self.middle:
            return f"{self.first} {self.middle}. {self.last}"
        return f"{self.first} {self.last}"

    @property
    def plain(self):
        return f"{self.first} {self.last}"


def fmt_date(rng, year, month, day):
    style = rng.random()
    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    mon_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                "Sep", "Oct", "Nov", "Dec"]
    if style < 0.3:
        return f"{month:02d}/{day:02d}/{year}"
    if style < 0.6:
        return f"{months[month - 1]} {day}, {year}"
    if style < 0.8:
        return f"{year}-{month:02d}-{day:02d}"
    return f"{mon_abbr[month - 1]} {day}, {year}"


def rand_date(rng, y0, y1):
    return rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28)


def phone(rng):
    area = rng.choice(AREA_CODES)
    line = f"01{rng.randint(10, 99)}"
    style = rng.random()
    if style < 0.4:
        return f"({area}) 555-{line}"
    if style < 0.7:
        return f"{area}-555-{line}"
    return f"{area}.555.{line}"


def ssn(rng):
    return f"{rng.randint(100, 665):03d}-{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}"


def zipcode(rng, prefix):
    if rng.random() < 0.25:
        return f"{prefix}{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
    return f"{prefix}{rng.randint(10, 99)}"


def address_block(nb, rng):
    """Street (LOCATION), city (LOCATION), state, ZIP (ZIP_CODE)."""
    city, state, zip_prefix = rng.choice(CITIES)
    street = f"{rng.randint(12, 4890)} {rng.choice(STREETS)}"
    nb.phi(street, "LOCATION")
    nb.add(", ").phi(city, "LOCATION").add(f", {state} ")
    nb.phi(zipcode(rng, zip_prefix), "ZIP_CODE")
    return nb


def contact_lines(nb, rng, patient, prob):
    if rng.random() < 0.9:
        nb.add("\nPhone: ").phi(phone(rng), "PHONE_NUMBER")
    if rng.random() < 0.45:
        nb.add("\nFax: ").phi(phone(rng), "FAX_NUMBER")
    if rng.random() < prob:
        domain = rng.choice(["examplemail.com", "stmary-example.org", "clinicpost-example.net"])
        email = f"{patient.first.lower()}.{patient.last.lower()}{rng.randint(2, 99)}@{domain}"
        nb.add("\nEmail: ").phi(email, "EMAIL_ADDRESS")


def id_lines(nb, rng, prob_account=0.55):
    nb.add("\nMRN: ").phi(f"{rng.randint(1000000, 9999999)}", "MEDICAL_RECORD_NUMBER")
    if rng.random() < prob_account:
        style = rng.random()
        if style < 0.5:
            nb.add("\nHealth Plan ID: ").phi(
                f"HP-{rng.randint(10000000, 99999999)}", "ACCOUNT_NUMBER")
        else:
            nb.add("\nAccount No: ").phi(
                f"{rng.randint(100000000, 999999999)}", "ACCOUNT_NUMBER")
    if rng.random() < 0.5:
        nb.add("\nSSN: ").phi(ssn(rng), "US_SSN")


def signature(nb, rng, doctor):
    nb.add("\n\nElectronically signed by Dr. ").phi(doctor.plain, "PERSON").add(", MD")
    if rng.random() < 0.45:
        nb.add("\nLicense #: ").phi(
            f"{rng.choice('ABCDEFGHJKLMNPRST')}{rng.randint(100000, 999999)}",
            "LICENSE_NUMBER")
    if rng.random() < 0.3:
        nb.add("\nDEA: ").phi(
            f"{rng.choice(FIRST_NAMES)[0]}{doctor.last[0]}{rng.randint(1000000, 9999999)}",
            "LICENSE_NUMBER")


def age_sentence(nb, rng, patient):
    """Age statement; only ages > 89 are labeled (Safe Harbor)."""
    age = rng.choice([61, 67, 72, 75, 78, 81, 84, 87, 90, 91, 92, 93, 95, 97, 101])
    style = rng.random()
    nb.add("\nThe patient is a")
    if style < 0.55:
        nb.add(" ")
        if age > 89:
            nb.phi(str(age), "AGE")
        else:
            nb.add(str(age))
        nb.add("-year-old ")
    elif style < 0.8:
        nb.add("n individual, age ")
        if age > 89:
            nb.phi(str(age), "AGE")
        else:
            nb.add(str(age))
        nb.add(", ")
    else:
        nb.add("n individual, aged ")
        if age > 89:
            nb.phi(str(age), "AGE")
        else:
            nb.add(str(age))
        nb.add(", ")
    nb.add(f"who presented with {rng.choice(CONDITIONS)}.")
    return nb


def discharge_summary(rng):
    patient, doctor = Person(rng), Person(rng)
    nb = NoteBuilder()
    nb.add("DISCHARGE SUMMARY\n")
    nb.add("Patient: ").phi(patient.full, "PERSON")
    nb.add("\nDOB: ").phi(fmt_date(rng, *rand_date(rng, 1935, 2005)), "DATE_TIME")
    id_lines(nb, rng)
    nb.add("\nAdmission Date: ").phi(fmt_date(rng, *rand_date(rng, 2024, 2026)), "DATE_TIME")
    nb.add("\nDischarge Date: ").phi(fmt_date(rng, *rand_date(rng, 2024, 2026)), "DATE_TIME")
    nb.add("\nAttending: Dr. ").phi(doctor.plain, "PERSON")
    nb.add("\nAddress: ")
    address_block(nb, rng)
    contact_lines(nb, rng, patient, 0.55)
    age_sentence(nb, rng, patient)
    nb.add(f"\nDuring this admission the patient {rng.choice(PROCEDURES)}.")
    nb.add(f"\n{patient.title} ").phi(patient.last, "PERSON")
    nb.add(" tolerated the procedure well and remained afebrile for 24 hours.")
    nb.add("\nDischarge medications include lisinopril 10 mg daily and aspirin 81 mg daily.")
    nb.add("\nFollow-up appointment scheduled for ").phi(
        fmt_date(rng, *rand_date(rng, 2026, 2026)), "DATE_TIME").add(" in the cardiology clinic.")
    if rng.random() < 0.3:
        nb.add("\nImplanted device interrogated. Device ID: ").phi(
            f"PM-{rng.randint(1000000, 9999999)}", "DEVICE_ID").add(", battery status satisfactory.")
    if rng.random() < 0.2:
        nb.add("\nResults are available on the patient portal at ").phi(
            "https://portal.stmary-example.org/results", "URL").add(".")
    signature(nb, rng, doctor)
    return "discharge_summary", nb


def referral_letter(rng):
    patient, doctor = Person(rng), Person(rng)
    nb = NoteBuilder()
    nb.add("REFERRAL LETTER\n")
    # the letter date is itself a date entity
    nb.add("Date: ").phi(fmt_date(rng, *rand_date(rng, 2025, 2026)), "DATE_TIME")
    nb.add("\n\nDear Dr. ").phi(doctor.plain, "PERSON").add(",\n")
    nb.add("\nI am referring my patient, ").phi(patient.plain, "PERSON")
    nb.add(" (DOB: ").phi(fmt_date(rng, *rand_date(rng, 1938, 2008)), "DATE_TIME").add("),")
    id_lines(nb, rng, prob_account=0.4)
    nb.add(f"\n{patient.title} ").phi(patient.last, "PERSON")
    nb.add(f" has been under my care since {rng.randint(2019, 2024)} for {rng.choice(CONDITIONS)}.")
    nb.add(" I would appreciate your evaluation and recommendations regarding further management.")
    nb.add("\nRecords may be faxed to our office. Fax: ").phi(phone(rng), "FAX_NUMBER")
    nb.add("\nOur office phone is ").phi(phone(rng), "PHONE_NUMBER").add(".")
    if rng.random() < 0.5:
        nb.add("\nThe patient can be reached at ").phi(phone(rng), "PHONE_NUMBER").add(".")
    if rng.random() < 0.25:
        nb.add("\nChart copies are posted at ").phi(
            "https://records.clinicpost-example.net/share", "URL").add(".")
    nb.add("\n\nSincerely,\nDr. ").phi(Person(rng).plain, "PERSON").add(", MD")
    nb.add("\nMailing address: ")
    address_block(nb, rng)
    return "referral_letter", nb


def progress_note(rng):
    patient, doctor = Person(rng), Person(rng)
    nb = NoteBuilder()
    nb.add("CLINIC PROGRESS NOTE\n")
    nb.add("Patient Name: ").phi(patient.full, "PERSON")
    nb.add("\nDate of Service: ").phi(fmt_date(rng, *rand_date(rng, 2025, 2026)), "DATE_TIME")
    id_lines(nb, rng, prob_account=0.45)
    age_sentence(nb, rng, patient)
    nb.add(f"\n{patient.title} ").phi(patient.last, "PERSON")
    nb.add(f" reports improved symptoms since the last visit and {rng.choice(PROCEDURES)} last month.")
    nb.add("\nVitals: BP 128/82, HR 74, Temp 98.4 F, SpO2 97 percent on room air.")
    nb.add("\nPlan: continue current medications, repeat labs in 3 months.")
    if rng.random() < 0.35:
        nb.add("\nInfusion pump in use. Serial No: ").phi(
            f"SN-{rng.randint(1000, 9999)}-{rng.randint(10000, 99999)}", "DEVICE_ID").add(".")
    if rng.random() < 0.1:
        nb.add("\nTransportation note: vehicle VIN: ").phi(
            "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(17)),
            "VEHICLE_ID").add(" recorded for the parking permit.")
    nb.add("\n\nSeen by Dr. ").phi(doctor.plain, "PERSON").add(", MD")
    return "progress_note", nb


def telehealth_note(rng):
    patient, doctor = Person(rng), Person(rng)
    nb = NoteBuilder()
    nb.add("TELEHEALTH VISIT NOTE\n")
    nb.add("Patient: ").phi(patient.plain, "PERSON")
    nb.add("\nVisit Date: ").phi(fmt_date(rng, *rand_date(rng, 2025, 2026)), "DATE_TIME")
    id_lines(nb, rng, prob_account=0.4)
    ip = ".".join(str(rng.choice([24, 73, 98, 104, 172, 198])) if i == 0
                  else str(rng.randint(0, 255)) for i in range(4))
    nb.add("\nThe video visit was conducted from the patient's home connection, IP address ")
    nb.phi(ip, "IP_ADDRESS").add(".")
    nb.add(f"\nThe patient reports symptoms consistent with {rng.choice(CONDITIONS)}.")
    nb.add("\nCallback number: ").phi(phone(rng), "PHONE_NUMBER")
    contact_lines(nb, rng, patient, 0.4)
    nb.add("\nHome address on file: ")
    address_block(nb, rng)
    nb.add("\n\nElectronically signed by Dr. ").phi(doctor.plain, "PERSON").add(", MD")
    return "telehealth_note", nb


GENERATORS = [discharge_summary, referral_letter, progress_note, telehealth_note]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "data" / "synthetic_notes.jsonl"),
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(args.count):
            note_type, nb = rng.choice(GENERATORS)(rng)
            record = {
                "id": f"syn-{i + 1:04d}",
                "note_type": note_type,
                "text": nb.build(),
                "entities": nb.entities,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.count} notes to {out_path}")


if __name__ == "__main__":
    main()
