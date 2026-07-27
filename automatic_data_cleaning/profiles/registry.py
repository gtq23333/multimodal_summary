from __future__ import annotations

from profiles.base import CleaningProfile
from profiles.graduate_competition import GraduateCompetitionProfile
from profiles.national_competition import NationalCompetitionProfile
from profiles.other_competition import OtherCompetitionProfile

_PROFILES: dict[str, type] = {
    "national_competition": NationalCompetitionProfile,
    "graduate_competition": GraduateCompetitionProfile,
    "other_competition": OtherCompetitionProfile,
}


def get_profile(name: str) -> CleaningProfile:
    cls = _PROFILES.get(name)
    if cls is None:
        known = ", ".join(sorted(_PROFILES))
        raise ValueError(f"未知 profile: {name}，可选: {known}")
    return cls()
