from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import jwt
import pytest

from xpeech.config.settings import JWTChannelConfig, settings
from xpeech.utils import channel_jwt
from xpeech.utils.channel_jwt import ChannelJWTAuth

KEY = "channel-object-test-signing-key-at-least-64-characters-long-for-all-hmac-algorithms"


def _token(**kwargs):
    return ChannelJWTAuth.create_token("crm", secret_key=KEY, **kwargs)


def test_request_jwt_is_decoded_using_the_configured_channel_key(monkeypatch):
    monkeypatch.setitem(settings.jwt.channels, "crm", JWTChannelConfig(secret_key=KEY))
    token = _token()
    auth = ChannelJWTAuth(token)
    assert auth.channel == "crm"
    assert set(auth.claims) == {"exp"}
    assert auth.token == token
    assert auth.secret_key == KEY
    assert not auth.is_expired
    assert token not in repr(auth)
    assert KEY not in repr(auth)
    assert ChannelJWTAuth(ChannelJWTAuth.create_token("crm")).channel == "crm"


def test_credential_properties_cannot_be_mutated_or_reissued_on_access():
    auth = ChannelJWTAuth(_token(), KEY)
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
    auth = ChannelJWTAuth(token, KEY)
    assert decode.call_count == 1
    for _ in range(3):
        assert auth.channel == "crm"
        assert auth.exp == int(auth.claims["exp"])
    assert decode.call_count == 1
    assert not auth.is_expired
    assert not auth.is_expired
    assert decode.call_count == 3


def test_credential_expiration_tracks_time_without_silently_replacing_the_token(monkeypatch):
    auth = ChannelJWTAuth(_token(), KEY)
    token = auth.token
    assert not auth.is_expired

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(auth.exp, UTC)

    monkeypatch.setattr(jwt.api_jwt, "datetime", Clock)
    assert auth.is_expired
    assert auth.token == token


def test_expired_credentials_are_rejected_at_initialization():
    token = _token(now=datetime.now(UTC) - timedelta(seconds=settings.jwt.access_token_expire_seconds + 1))
    with pytest.raises(jwt.ExpiredSignatureError):
        ChannelJWTAuth(token, KEY)


def test_expiration_property_follows_pyjwt_numeric_date_conversion():
    expires_at = int((datetime.now(UTC) + timedelta(seconds=60)).timestamp())
    token = jwt.encode(
        {"exp": str(expires_at)}, KEY,
        algorithm=settings.jwt.algorithm, headers={"kid": "crm"},
    )
    auth = ChannelJWTAuth(token, KEY)
    assert auth.exp == expires_at
    assert not auth.is_expired


def test_static_issuance_keeps_optional_claims_isolated():
    feishu_claim = ChannelJWTAuth(_token(lark_user_access_token="user-access-token"), KEY)
    plain = ChannelJWTAuth(_token(), KEY)
    assert set(plain.claims) == {"exp"}
    assert feishu_claim.claims["lark_user_access_token"] == "user-access-token"


def test_invalid_signature_and_unknown_channel_are_rejected():
    token = _token()
    with pytest.raises(jwt.InvalidTokenError):
        ChannelJWTAuth(token, "a-different-signing-key-at-least-32-characters")
    token = jwt.encode(
        {"exp": 9999999999}, KEY,
        algorithm=settings.jwt.algorithm, headers={"kid": "unknown-channel"},
    )
    with pytest.raises(jwt.InvalidTokenError):
        ChannelJWTAuth(token)
