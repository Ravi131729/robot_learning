"""Top-level robot policy composition."""

import jax

from configs.robot_policy_config import DIT_DEPTH
from robot_policy.action.embedding import action_embed, init_action_params
from robot_policy.conditioning.condition import (
    condition_embed,
    init_condition_params,
)
from robot_policy.conditioning.timestep import init_time_params
from robot_policy.dit.transformer import dit_forward, init_dit_params
from robot_policy.vision.encoder import encode_vision_for_dit, init_vision_params


def init_policy_params(key, dit_depth=DIT_DEPTH):
    """Initialize all parameter groups used by the robot policy."""
    vision_key, time_key, condition_key, action_key, dit_key = jax.random.split(
        key, 5
    )

    return {
        "vision": init_vision_params(vision_key),
        "time": init_time_params(time_key),
        "condition": init_condition_params(condition_key),
        "action": init_action_params(action_key),
        "dit": init_dit_params(dit_key, depth=dit_depth),
    }


def policy_forward(dino_tokens, state, task, noisy_actions, t, params):
    """Predict action velocity from visual, state, task, and time inputs."""
    vision = encode_vision_for_dit(dino_tokens, params["vision"])
    condition = condition_embed(
        state,
        task,
        t,
        params["condition"],
        params["time"],
    )
    action_tokens = action_embed(noisy_actions, params["action"])

    return dit_forward(
        action_tokens,
        vision,
        condition,
        params["dit"],
    )


# Compatibility names for existing ABC-oriented experiment code.
init_abc_params = init_policy_params
abc_forward = policy_forward
