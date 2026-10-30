import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from phi_shield.deidentify import Deidentifier  # noqa: E402


@pytest.fixture(scope="session")
def deid():
    return Deidentifier()
