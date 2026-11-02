from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import jwt
import pytest

from xpeech.config.settings import JWTChannelConfig, settings
from xpeech.utils import channel_jwt
from xpeech.utils.channel_jwt import ChannelJWTAuth, ChannelTokenParams, create_token

KEY = "channel-object-test-signing-key-at-least-64-characters-long-for-all-hmac-algorithms"


@pytest.fixture(autouse=True)
def configured_channel(monkeypatch):
    monkeypatch.setitem(settings.jwt.channels, "crm", JWTChannelConfig(secret_key=KEY))


def _token(**kwargs):
    return create_token(ChannelTokenParams("crm", **kwargs))


def test_request_jwt_is_decoded_using_the_configured_channel_key():
    token = _token()
    auth = ChannelJWTAuth(token)
    assert auth.channel == "crm"
    assert set(auth.claims) == {"exp"}
    assert auth.token == token
    assert auth.secret_key == KEY
    assert token not in repr(auth)
    assert KEY not in repr(auth)
    assert ChannelJWTAuth(create_token(ChannelTokenParams("crm"))).channel == "crm"


def test_token_issuance_verifies_the_supplied_channel_secret():
    token = create_token(ChannelTokenParams("crm"), secret_key4verify=KEY)
    auth = ChannelJWTAuth(token)
    assert auth.channel == "crm"
    assert set(auth.claims) == {"exp"}


@pytest.mark.parametrize("secret_key", ["", "wrong-channel-secret"])
def test_token_issuance_rejects_an_incorrect_secret_before_signing(monkeypatch, secret_key):
    encode = Mock(wraps=jwt.encode)
    monkeypatch.setattr(channel_jwt.jwt, "encode", encode)
    with pytest.raises(jwt.InvalidTokenError, match="Incorrect JWT secret"):
        create_token(ChannelTokenParams("crm"), secret_key4verify=secret_key)
    encode.assert_not_called()


def test_credential_properties_cannot_be_mutated_or_reissued_on_access():
    auth = ChannelJWTAuth(_token())
    token = auth.token
    for name, value in (("channel", "erp"), ("exp", 0), ("token", "replacement"), ("secret_key", "other")):
        with pytest.raises(AttributeError):
            setattr(auth, name, value)
    with pytest.raises(TypeError):
        auth.claims["exp"] = 0
    assert auth.token == token
    assert auth.channel == "crm"


def test_identity_is_decoded_at_initialization_and_properties_reuse_it(monkeypatch):
    token = _token()
    decode = Mock(wraps=jwt.decode)
    monkeypatch.setattr(channel_jwt.jwt, "decode", decode)
    auth = ChannelJWTAuth(token)
    assert decode.call_count == 1
    for _ in range(3):
        assert auth.channel == "crm"
        assert auth.exp == int(auth.claims["exp"])
    assert decode.call_count == 1


def test_expired_credentials_are_rejected_at_initialization():
    token = jwt.encode(
        {"exp": datetime.now(UTC) - timedelta(seconds=1)}, KEY,
        algorithm=settings.jwt.algorithm, headers={"kid": "crm"},
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        ChannelJWTAuth(token)


def test_expiration_property_follows_pyjwt_numeric_date_conversion():
    expires_at = int((datetime.now(UTC) + timedelta(seconds=60)).timestamp())
    token = jwt.encode(
        {"exp": str(expires_at)}, KEY,
        algorithm=settings.jwt.algorithm, headers={"kid": "crm"},
    )
    auth = ChannelJWTAuth(token)
    assert auth.exp == expires_at


def test_token_issuance_keeps_optional_claims_isolated():
    feishu_claim = ChannelJWTAuth(_token(lark_user_access_token="user-access-token"))
    plain = ChannelJWTAuth(_token())
    assert set(plain.claims) == {"exp"}
    assert feishu_claim.claims["lark_user_access_token"] == "user-access-token"


def test_invalid_signature_and_unknown_channel_are_rejected():
    token = jwt.encode(
        {"exp": 9999999999}, "a-different-signing-key-at-least-32-characters",
        algorithm=settings.jwt.algorithm, headers={"kid": "crm"},
    )
    with pytest.raises(jwt.InvalidTokenError):
        ChannelJWTAuth(token)
    token = jwt.encode(
        {"exp": 9999999999}, KEY,
        algorithm=settings.jwt.algorithm, headers={"kid": "unknown-channel"},
    )
    with pytest.raises(jwt.InvalidTokenError):
        ChannelJWTAuth(token)
