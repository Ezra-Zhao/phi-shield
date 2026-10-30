"""PHI-Shield: clinical-text PHI detection & de-identification (HIPAA Safe Harbor style).

Built on Microsoft Presidio (analyzer + anonymizer) with custom recognizers
for clinical identifiers (MRN, account / health-plan numbers, device IDs,
licenses, VINs, ZIP codes, fax numbers, ages over 89, street addresses) and
title-anchored name patterns that back up spaCy NER.
"""

__all__ = ["build_analyzer", "Deidentifier", "SUPPORTED_ENTITIES"]

from .engine import build_analyzer, SUPPORTED_ENTITIES  # noqa: E402
from .deidentify import Deidentifier  # noqa: E402
