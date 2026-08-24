from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .dao import User


SESSION_ID_PATTERN = r"^[\w@+-][\w.@+-]*$"


@dataclass(frozen=True)
class OAuth2WebConfig:
    provider_name: str
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    userinfo_url: str
    redirect_uri: str | None
    scopes: tuple[str, ...]
    session_id_claim: str
    username_claim: str
    use_pkce: bool
    token_auth_method: str
    extra_authorization_params: dict[str, str]
    auto_create_users: bool = False
    display_type: str = "qrcode"


@dataclass(frozen=True)
class WebConfig:
    backend_url: str
    database_path: Path
    static_dir: Path
    system_name: str
    cookie_name: str = "xpeech_session"
    oauth2: OAuth2WebConfig | None = None


class LoginBody(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)


class PasswordChangeBody(BaseModel):
    new_password: str = Field(min_length=8, max_length=256)


class OAuth2PollBody(BaseModel):
    login_id: str = Field(min_length=1, max_length=128)
    poll_token: str = Field(min_length=1, max_length=256)


class UserBody(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=SESSION_ID_PATTERN,
    )
    username: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@+-]+$")
    password: str = Field(min_length=8, max_length=256)
    is_admin: bool = False


class UserUpdateBody(BaseModel):
    username: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[\w.@+-]+$",
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=SESSION_ID_PATTERN,
    )
    password: str | None = Field(default=None, min_length=8, max_length=256)
    is_admin: bool | None = None
    is_active: bool | None = None


def public_user(user: User) -> dict[str, object]:
    return {
        "id": user.id,
        "session_id": user.session_id,
        "username": user.username,
        "is_admin": user.is_admin,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat(),
    }
