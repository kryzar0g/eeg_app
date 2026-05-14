from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import re
import unicodedata
from pathlib import Path
from typing import List

from .config import PROJECT_ROOT

PROFILE_DIR = PROJECT_ROOT / "data" / "patients"


@dataclass
class PatientProfile:
    profile_id: str
    first_name: str
    last_name: str
    date_of_birth: str
    sex: str
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def display_name(self) -> str:
        return f"{self.last_name}, {self.first_name} ({self.date_of_birth})"


def _ensure_dir() -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "patient"


def build_profile_id(first_name: str, last_name: str, date_of_birth: str) -> str:
    return "_".join([
        _slugify(last_name),
        _slugify(first_name),
        _slugify(date_of_birth),
    ])


def profile_path(profile_id: str) -> Path:
    _ensure_dir()
    return PROFILE_DIR / f"{profile_id}.json"


def save_profile(profile: PatientProfile) -> PatientProfile:
    _ensure_dir()
    now = datetime.now().isoformat(timespec="seconds")
    data = asdict(profile)
    if not data.get("created_at"):
        data["created_at"] = now
    data["updated_at"] = now
    profile = PatientProfile(**data)

    with profile_path(profile.profile_id).open("w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

    return profile


def load_profile(profile_id: str) -> PatientProfile:
    path = profile_path(profile_id)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return PatientProfile(**data)


def list_profiles() -> List[PatientProfile]:
    _ensure_dir()
    profiles: List[PatientProfile] = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            profiles.append(PatientProfile(**data))
        except Exception:
            continue
    profiles.sort(key=lambda profile: (profile.last_name.lower(), profile.first_name.lower(), profile.date_of_birth))
    return profiles


def create_profile(first_name: str, last_name: str, date_of_birth: str, sex: str, notes: str = "") -> PatientProfile:
    profile = PatientProfile(
        profile_id=build_profile_id(first_name, last_name, date_of_birth),
        first_name=first_name.strip(),
        last_name=last_name.strip(),
        date_of_birth=date_of_birth.strip(),
        sex=sex.strip(),
        notes=notes.strip(),
    )
    return save_profile(profile)
