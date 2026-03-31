"""
models_db.py — SQLAlchemy Database Models
==========================================
User accounts (Google OAuth) and saved analysis results.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id          = db.Column(db.Integer, primary_key=True)
    google_id   = db.Column(db.String(120), unique=True, nullable=False)
    name        = db.Column(db.String(200), nullable=False)
    email       = db.Column(db.String(200), unique=True, nullable=False)
    profile_pic = db.Column(db.String(500), default="")
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship
    results = db.relationship("AnalysisResult", backref="user", lazy=True,
                              order_by="AnalysisResult.created_at.desc()")

    def __repr__(self):
        return f"<User {self.name} ({self.email})>"


class AnalysisResult(db.Model):
    __tablename__ = "analysis_results"

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    job_id      = db.Column(db.String(20), nullable=False)
    mode        = db.Column(db.String(20), default="single")  # single / multi_view

    # Scores
    stability   = db.Column(db.Float, default=0.0)
    efficiency  = db.Column(db.Float, default=0.0)
    precision   = db.Column(db.Float, default=0.0)
    smoothness  = db.Column(db.Float, default=0.0)
    overall     = db.Column(db.Float, default=0.0)
    grade       = db.Column(db.String(5), default="F")

    # Extra metrics
    tracking_quality = db.Column(db.Float, default=0.0)
    frames           = db.Column(db.Integer, default=0)

    # AI feedback
    ai_comment  = db.Column(db.Text, default="")

    # Timestamps
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AnalysisResult {self.job_id} ({self.overall:.1f})>"

    def to_dict(self):
        return {
            "id":          self.id,
            "job_id":      self.job_id,
            "mode":        self.mode,
            "stability":   self.stability,
            "efficiency":  self.efficiency,
            "precision":   self.precision,
            "smoothness":  self.smoothness,
            "overall":     self.overall,
            "grade":       self.grade,
            "tracking_quality": self.tracking_quality,
            "frames":      self.frames,
            "ai_comment":  self.ai_comment,
            "created_at":  self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "",
        }
