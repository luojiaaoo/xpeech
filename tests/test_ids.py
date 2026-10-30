from uuid import UUID

import pytest

from xpeech.exceptions import PathProtectionError
from xpeech.utils.ids import generate_session_id, validate_session_id


def test_session_id_accepts_default_and_canonical_uuidv7():
    uuid7 = "018f0000-0000-7000-8000-000000000001"

    assert validate_session_id("default") == "default"
    assert validate_session_id(uuid7) == uuid7


@pytest.mark.parametrize(
    "session_id",
    [
        "chat-1",
        "01234567-89ab-4def-8123-456789abcdef",
        "018F0000-0000-7000-8000-000000000001",
        "018f0000000070008000000000000001",
    ],
)
def test_session_id_rejects_non_uuidv7_or_noncanonical_values(session_id):
    with pytest.raises(PathProtectionError, match="UUIDv7"):
        validate_session_id(session_id)


def test_generate_session_id_returns_unique_canonical_uuidv7_values():
    generated = {generate_session_id() for _ in range(100)}

    assert len(generated) == 100
    for value in generated:
        parsed = UUID(value)
        assert parsed.version == 7
        assert str(parsed) == value
