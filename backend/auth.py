"""Who is calling. STUB: reads two headers instead of a Supabase Auth JWT.

    X-User-Id: <uuid>        X-User-Role: viewer | brand_partner | stylist | admin
    X-Brand-Id: <uuid>       (brand partners only)

Milestone 3: verify the Supabase JWT (OAuth 2.0: email, Google, Apple) and read the role
from the users table. Scanning works without an account, so `current_user` may be None.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from .schemas import Role, User


def current_user(x_user_id: str | None = Header(None), x_user_role: str | None = Header(None),
                 x_brand_id: str | None = Header(None)) -> User | None:
    if not x_user_id:
        return None
    try:
        role = Role(x_user_role or "viewer")
    except ValueError:
        raise HTTPException(400, "unknown role")
    return User(id=x_user_id, role=role, brand_id=x_brand_id)


def require_login(user: User | None = Depends(current_user)) -> User:
    if user is None:
        raise HTTPException(401, "login required")
    return user


def require_role(*roles: Role):
    def check(user: User = Depends(require_login)) -> User:
        if user.role not in roles:
            raise HTTPException(403, "not allowed for this role")
        return user
    return check
