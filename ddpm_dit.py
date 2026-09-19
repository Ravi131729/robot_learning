import jax
import jax.numpy as jnp

from configs.robot_policy_config import (
    ACTION_DIM,
    ACTION_HORIZON,
    DINO_DIM,
    DIT_DEPTH,
    DIT_DIM,
    DIT_HEADS,
    DIT_MLP_DIM,
    NUM_CAMERAS,
    NUM_POOL_HEADS,
    NUM_POOL_QUERIES,
    NUM_VISION_TOKENS,
    POOL_MLP_DIM,
    STATE_DIM,
    TASK_DIM,
    TIME_EMBED_DIM,
)
from robot_policy.common.attention import cross_attention, self_attention
from robot_policy.common.layers import init_weight, layer_norm, modulate
from robot_policy.action.embedding import (
    action_embed,
    init_action_params,
    positional_embedding,
)
from robot_policy.conditioning.condition import (
    condition_embed,
    init_condition_params,
)
from robot_policy.conditioning.timestep import (
    init_time_params,
    time_embed,
    timestep_embedding,
)
from robot_policy.dit.block import dit_block, init_dit_block
from robot_policy.dit.output import final_layer
from robot_policy.dit.transformer import dit_forward, init_dit_params
from robot_policy.policy import (
    abc_forward,
    init_abc_params,
    init_policy_params,
    policy_forward,
)
from robot_policy.objectives.flow_matching import flow_matching_loss
from robot_policy.sampling.flow_sampler import sample_actions
from robot_policy.vision.encoder import (
    encode_camera,
    encode_vision_for_dit,
    init_vision_params,
)
from robot_policy.vision.pooler import cross_attention_pool, init_pooler, vision_pool_block


# ============================================================
# Small test
# ============================================================

if __name__ == "__main__":
    # Use smaller depth while debugging
    B = 2

    key = jax.random.PRNGKey(0)
    kp, kd, ks, kt, ka, kl = jax.random.split(key, 6)

    print("Initializing model...")
    params = init_abc_params(kp)

    # Fake DINO output
    dino_tokens = jax.random.normal(
        kd,
        (B, NUM_VISION_TOKENS, NUM_CAMERAS, DINO_DIM),
    )

    # Robot state
    state = jax.random.normal(
        ks,
        (B, STATE_DIM),
    )

    # Fake CLIP task embedding
    task = jax.random.normal(
        kt,
        (B, TASK_DIM),
    )

    # Ground-truth action chunk
    actions = jax.random.normal(
        ka,
        (B, ACTION_HORIZON, ACTION_DIM),
    )

    loss = flow_matching_loss(
        params,
        dino_tokens,
        state,
        task,
        actions,
        kl,
    )

    print("DINO:", dino_tokens.shape)
    print("State:", state.shape)
    print("Task:", task.shape)
    print("Actions:", actions.shape)
    print("Loss:", loss)
