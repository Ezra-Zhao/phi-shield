# PHI-Shield evaluation report

Dataset: 200 synthetic clinical notes (data/synthetic_notes.jsonl). All data synthetic; no real patient data.
Analyzer backend: spacy:en_core_web_sm

Matching rule: same entity type and predicted span covers >= 50% of the gold span; greedy one-to-one matching.

**Overall precision: 0.945 | Overall recall: 0.997 | Overall F1: 0.970**

| Entity type | Gold | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| ACCOUNT_NUMBER | 76 | 76 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| AGE | 54 | 54 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| DATE_TIME | 387 | 386 | 56 | 1 | 0.873 | 0.997 | 0.931 |
| DEVICE_ID | 35 | 35 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| EMAIL_ADDRESS | 48 | 48 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| FAX_NUMBER | 87 | 87 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| IP_ADDRESS | 44 | 44 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| LICENSE_NUMBER | 27 | 27 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| LOCATION | 286 | 282 | 6 | 4 | 0.979 | 0.986 | 0.983 |
| MEDICAL_RECORD_NUMBER | 200 | 200 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| PERSON | 655 | 654 | 75 | 1 | 0.897 | 0.998 | 0.945 |
| PHONE_NUMBER | 206 | 206 | 2 | 0 | 0.990 | 1.000 | 0.995 |
| URL | 21 | 21 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| US_SSN | 101 | 101 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| VEHICLE_ID | 3 | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| ZIP_CODE | 143 | 143 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| **OVERALL** | 2373 | 2367 | 139 | 6 | 0.945 | 0.997 | 0.970 |

