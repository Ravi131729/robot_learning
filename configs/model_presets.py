"""Named architecture presets for development and training."""

from dataclasses import dataclass

from configs.robot_policy_config import DIT_DEPTH


@dataclass(frozen=True)
class RobotPolicyPreset:
    """Static architecture choices passed to parameter initialization."""

    name: str
    dit_depth: int


DEBUG_PRESET = RobotPolicyPreset(
    name="debug",
    dit_depth=1,
)

SMALL_PRESET = RobotPolicyPreset(
    name="small",
    dit_depth=8,
)

FULL_PRESET = RobotPolicyPreset(
    name="full",
    dit_depth=DIT_DEPTH,
)


PRESETS = {
    DEBUG_PRESET.name: DEBUG_PRESET,
    SMALL_PRESET.name: SMALL_PRESET,
    FULL_PRESET.name: FULL_PRESET,
}


def get_preset(name):
    """Return a named architecture preset."""
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model preset {name!r}; choose from {tuple(PRESETS)}"
        ) from exc
