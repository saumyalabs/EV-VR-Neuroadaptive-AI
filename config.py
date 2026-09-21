"""Shared column definitions for legacy thesis scripts."""

from __future__ import annotations

IDENTIFIER_COLUMNS = [
    "participant_id",
    "participant",
    "station_id",
    "station",
    "trial",
    "group",
    "threat",
    "threat_framing",
    "ai",
    "ai_available",
    "behaviour_token",
    "timestamp",
    "window_id",
]

TARGET_COLUMNS = [
    "ai_used",
    "ai_followed",
    "unsafe_choice",
    "ai_usage_increased",
    "ai_usage_decreased",
]
