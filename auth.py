"""
auth.py — Google OAuth 2.0 Authentication
===========================================
Blueprint for login, logout, and Google OAuth callback.
"""

import os
import json
from functools import wraps
from flask import Blueprint, redirect, url_for, session, request, render_template, flash
from authlib.integrations.flask_client import OAuth
from models_db import db, User

auth_bp = Blueprint("auth", __name__)

# ── OAuth setup (called from app.py) ──────────────────────────────────────────
oauth = OAuth()


def init_oauth(app):
    """Initialize OAuth with the Flask app."""
    oauth.init_app(app)
    oauth.register(
        name="google",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def login_required(f):
    """Decorator to protect routes — redirects to login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            # Store the intended URL so we can redirect after login
            session["next_url"] = request.url
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


def get_current_user():
    """Return the current User model instance or None."""
    if "user" not in session:
        return None
    return User.query.filter_by(google_id=session["user"]["google_id"]).first()


# ── Routes ────────────────────────────────────────────────────────────────────

@auth_bp.route("/login")
def login():
    """Show login page or start OAuth flow."""
    if "user" in session:
        return redirect(url_for("my_dashboard"))
    return render_template("login.html")


@auth_bp.route("/login/google")
def login_google():
    """Initiate Google OAuth flow."""
    redirect_uri = url_for("auth.google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)


@auth_bp.route("/login/callback")
def google_callback():
    """Handle Google OAuth callback."""
    try:
        token = oauth.google.authorize_access_token()
        user_info = token.get("userinfo")

        if not user_info:
            # Fallback: fetch from userinfo endpoint
            resp = oauth.google.get("https://openidconnect.googleapis.com/v1/userinfo")
            user_info = resp.json()

        # Find or create user
        user = User.query.filter_by(google_id=user_info["sub"]).first()
        if not user:
            user = User(
                google_id=user_info["sub"],
                name=user_info.get("name", "User"),
                email=user_info.get("email", ""),
                profile_pic=user_info.get("picture", ""),
            )
            db.session.add(user)
            db.session.commit()
        else:
            # Update profile info on each login
            user.name = user_info.get("name", user.name)
            user.profile_pic = user_info.get("picture", user.profile_pic)
            db.session.commit()

        # Store in session
        session["user"] = {
            "google_id":   user.google_id,
            "name":        user.name,
            "email":       user.email,
            "profile_pic": user.profile_pic,
        }

        # Redirect to intended page or dashboard
        next_url = session.pop("next_url", None)
        if next_url:
            return redirect(next_url)
        return redirect(url_for("my_dashboard"))

    except Exception as e:
        print(f"[Auth] OAuth error: {e}")
        return redirect(url_for("auth.login"))


@auth_bp.route("/logout")
def logout():
    """Clear session and redirect to home."""
    session.pop("user", None)
    return redirect(url_for("index"))
