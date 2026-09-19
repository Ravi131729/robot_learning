import jax
import jax.numpy as jnp


# Config
NUM_VISION_TOKENS = 197
DINO_DIM = 768
NUM_POOL_QUERIES = 12
NUM_POOL_HEADS = 8
POOL_MLP_DIM = 4 * DINO_DIM
NUM_CAMERAS = 3

DIT_DIM = 1536
DIT_HEADS = 24
DIT_DEPTH = 32
DIT_MLP_DIM = 4 * DIT_DIM

STATE_DIM = 14
TASK_DIM = 512
ACTION_DIM = 14
ACTION_HORIZON = 30
TIME_EMBED_DIM = 256


# Basic ops
def init_weight(key, in_dim, out_dim, scale=0.02):
    return scale * jax.random.normal(key, (in_dim, out_dim))


def layer_norm(x, scale=None, bias=None, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    y = (x - mean) / jnp.sqrt(var + eps)
    if scale is not None:
        y = y * scale
    if bias is not None:
        y = y + bias
    return y


def modulate(x, shift, scale):
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


# ============================================================
# Vision pooler
# ============================================================

def init_pooler(key):
    k = jax.random.split(key, 7)

    return {
        "pool_queries": 0.02 * jax.random.normal(
            k[0], (NUM_POOL_QUERIES, DINO_DIM)
        ),

        "qkv_weight": init_weight(k[1], DINO_DIM, 3 * DINO_DIM),
        "qkv_bias": jnp.zeros(3 * DINO_DIM),
        "out_weight": init_weight(k[4], DINO_DIM, DINO_DIM),
        "out_bias": jnp.zeros(DINO_DIM),

        "ffn_up": init_weight(k[5], DINO_DIM, POOL_MLP_DIM),
        "ffn_up_bias": jnp.zeros(POOL_MLP_DIM),

        "ffn_down": init_weight(k[6], POOL_MLP_DIM, DINO_DIM),
        "ffn_down_bias": jnp.zeros(DINO_DIM),

        "pool_ln_scale": jnp.ones(DINO_DIM),
        "pool_ln_bias": jnp.zeros(DINO_DIM),
        "ffn_ln_scale": jnp.ones(DINO_DIM),
        "ffn_ln_bias": jnp.zeros(DINO_DIM),
    }


def init_vision_params(key):
    k = jax.random.split(key, 5)

    return {
        "poolers": {
            "top": init_pooler(k[0]),
            "left": init_pooler(k[1]),
            "right": init_pooler(k[2]),
        },

        "vision_to_dit": init_weight(k[3], DINO_DIM, DIT_DIM),
        "vision_to_dit_bias": jnp.zeros(DIT_DIM),
        # Retained for exact official state-tree coverage; official forward
        # keeps this compatibility projection unused.
        "img_proj": init_weight(k[3], DINO_DIM, DIT_DIM),
        "img_proj_bias": jnp.zeros(DIT_DIM),

        "camera_embed": 0.02 * jax.random.normal(
            k[4], (NUM_CAMERAS, DIT_DIM)
        ),
    }


def cross_attention_pool(vision_tokens, p):
    B, T, D = vision_tokens.shape
    H = NUM_POOL_HEADS
    Dh = D // H

    vt = layer_norm(vision_tokens, p["pool_ln_scale"], p["pool_ln_bias"], eps=1e-5)
    pq = layer_norm(p["pool_queries"], p["pool_ln_scale"], p["pool_ln_bias"], eps=1e-5)
    qw, kw, vw = jnp.split(p["qkv_weight"], 3, axis=-1)
    qb, kb, vb = jnp.split(p["qkv_bias"], 3, axis=-1)
    q = (pq @ qw + qb).reshape(1, NUM_POOL_QUERIES, H, Dh).transpose(0, 2, 1, 3)
    k = (vt @ kw + kb).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    v = (vt @ vw + vb).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2)
    scores = scores / jnp.sqrt(Dh)

    attn = jax.nn.softmax(scores, axis=-1)

    x = (attn @ v).transpose(
        0, 2, 1, 3
    ).reshape(B, NUM_POOL_QUERIES, D)

    return x @ p["out_weight"] + p["out_bias"]


def vision_pool_block(vision_tokens, p):
    x = cross_attention_pool(vision_tokens, p)
    residual = x

    # FFN
    x = layer_norm(x, p["ffn_ln_scale"], p["ffn_ln_bias"], eps=1e-5)

    x = jax.nn.gelu(
        x @ p["ffn_up"] + p["ffn_up_bias"]
    )

    x = x @ p["ffn_down"] + p["ffn_down_bias"]

    return residual + x


def encode_camera(dino_tokens, camera_name, camera_id, p):
    # Camera-specific pooler
    x = vision_pool_block(
        dino_tokens,
        p["poolers"][camera_name],
    )

    # 768 -> 1536
    x = x @ p["vision_to_dit"] + p["vision_to_dit_bias"]

    # Camera identity
    return x + p["camera_embed"][camera_id]


def encode_vision_for_dit(dino_tokens, p):
    names = ("top", "left", "right")

    views = [
        encode_camera(
            dino_tokens[:, :, i, :],
            name,
            i,
            p,
        )
        for i, name in enumerate(names)
    ]

    # (B,36,1536)
    return jnp.concatenate(views, axis=1)


# ============================================================
# Timestep
# ============================================================

def init_time_params(key):
    k1, k2 = jax.random.split(key)

    return {
        "w1": init_weight(
            k1, TIME_EMBED_DIM, DIT_DIM
        ),
        "b1": jnp.zeros(DIT_DIM),

        "w2": init_weight(
            k2, DIT_DIM, DIT_DIM
        ),
        "b2": jnp.zeros(DIT_DIM),
    }


def timestep_embedding(t, dim=TIME_EMBED_DIM):
    half = dim // 2

    freqs = jnp.exp(
        -jnp.log(10000.0)
        * jnp.arange(half)
        / half
    )

    args = t[:, None] * freqs[None, :]

    return jnp.concatenate(
        [jnp.cos(args), jnp.sin(args)],
        axis=-1,
    )


def time_embed(t, p):
    # (B,) -> (B,256) -> (B,1536)
    x = timestep_embedding(t)

    x = jax.nn.silu(
        x @ p["w1"] + p["b1"]
    )

    return x @ p["w2"] + p["b2"]


# ============================================================
# Conditioning
# ============================================================

def init_condition_params(key):
    k = jax.random.split(key, 4)

    return {
        "state_w": init_weight(
            k[0], STATE_DIM, DIT_DIM
        ),
        "state_b": jnp.zeros(DIT_DIM),

        "task_w": init_weight(
            k[1], TASK_DIM, DIT_DIM
        ),
        "task_b": jnp.zeros(DIT_DIM),

        "cond_w1": init_weight(
            k[2], 3 * DIT_DIM, DIT_DIM
        ),
        "cond_b1": jnp.zeros(DIT_DIM),

        "cond_w2": init_weight(
            k[3], DIT_DIM, DIT_DIM
        ),
        "cond_b2": jnp.zeros(DIT_DIM),

        "cond_ln_scale": jnp.ones(DIT_DIM),
        "cond_ln_bias": jnp.zeros(DIT_DIM),
    }


def condition_embed(state, task, t, p, time_p):
    # State and task
    state_emb = state @ p["state_w"] + p["state_b"]
    task_emb = task @ p["task_w"] + p["task_b"]
    time_emb = time_embed(t, time_p)

    # (B,4608)
    x = jnp.concatenate(
        [state_emb, task_emb, time_emb],
        axis=-1,
    )

    x = jax.nn.silu(
        x @ p["cond_w1"] + p["cond_b1"]
    )

    x = x @ p["cond_w2"] + p["cond_b2"]

    return layer_norm(x, p["cond_ln_scale"], p["cond_ln_bias"], eps=1e-5)


# ============================================================
# Action embedding
# ============================================================

def init_action_params(key):
    return {
        "w": init_weight(
            key, ACTION_DIM, DIT_DIM
        ),
        "b": jnp.zeros(DIT_DIM),
        # Official pos_embed is a non-trainable (1, 30, 1536) parameter.
        "pos_embed": positional_embedding(ACTION_HORIZON, DIT_DIM)[None, :, :],
    }


def positional_embedding(length, dim):
    half = dim // 2

    freqs = jnp.exp(
        -jnp.log(10000.0)
        * jnp.arange(half)
        / half
    )

    pos = jnp.arange(length)[:, None]
    args = pos * freqs[None, :]

    return jnp.concatenate(
        [jnp.sin(args), jnp.cos(args)],
        axis=-1,
    )


def action_embed(actions, p):
    # (B,30,14) -> (B,30,1536)
    x = actions @ p["w"] + p["b"]

    return x + p["pos_embed"][:, : actions.shape[1], :]


# ============================================================
# DiT attention
# ============================================================

def self_attention(x, p):
    B, T, D = x.shape
    H = DIT_HEADS
    Dh = D // H

    qw, kw, vw = jnp.split(p["sa_qkv"], 3, axis=-1)
    qb, kb, vb = jnp.split(p["sa_qkv_bias"], 3, axis=-1)
    q = (x @ qw + qb).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    k = (x @ kw + kb).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    v = (x @ vw + vb).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2)
    scores = scores / jnp.sqrt(Dh)

    attn = jax.nn.softmax(
        scores,
        axis=-1,
    )

    x = (attn @ v).transpose(
        0, 2, 1, 3
    ).reshape(B, T, D)

    return x @ p["sa_out"] + p["sa_out_bias"]


def cross_attention(x, vision, p):
    B, T, D = x.shape
    V = vision.shape[1]

    H = DIT_HEADS
    Dh = D // H

    qw, kw, vw = jnp.split(p["ca_qkv"], 3, axis=-1)
    qb, kb, vb = jnp.split(p["ca_qkv_bias"], 3, axis=-1)
    q = (x @ qw + qb).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    k = (vision @ kw + kb).reshape(B, V, H, Dh).transpose(0, 2, 1, 3)
    v = (vision @ vw + vb).reshape(B, V, H, Dh).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2)
    scores = scores / jnp.sqrt(Dh)

    attn = jax.nn.softmax(
        scores,
        axis=-1,
    )

    x = (attn @ v).transpose(
        0, 2, 1, 3
    ).reshape(B, T, D)

    return x @ p["ca_out"] + p["ca_out_bias"]


# ============================================================
# DiT block
# ============================================================

def init_dit_block(key):
    k = jax.random.split(key, 11)

    return {
        # 9 AdaLN vectors
        "ada_w": init_weight(
            k[0], DIT_DIM, 9 * DIT_DIM
        ),
        "ada_b": jnp.zeros(9 * DIT_DIM),

        # PyTorch attention uses packed projections and explicit biases.
        "sa_qkv": init_weight(k[1], DIT_DIM, 3 * DIT_DIM),
        "sa_qkv_bias": jnp.zeros(3 * DIT_DIM),
        "sa_out": init_weight(k[4], DIT_DIM, DIT_DIM),
        "sa_out_bias": jnp.zeros(DIT_DIM),
        "ca_qkv": init_weight(k[5], DIT_DIM, 3 * DIT_DIM),
        "ca_qkv_bias": jnp.zeros(3 * DIT_DIM),
        "ca_out": init_weight(k[8], DIT_DIM, DIT_DIM),
        "ca_out_bias": jnp.zeros(DIT_DIM),

        # MLP
        "mlp_up": init_weight(
            k[9], DIT_DIM, DIT_MLP_DIM
        ),
        "mlp_up_bias": jnp.zeros(DIT_MLP_DIM),
        "mlp_down": init_weight(
            k[10], DIT_MLP_DIM, DIT_DIM
        ),
        "mlp_down_bias": jnp.zeros(DIT_DIM),

    }


def dit_block(x, vision, cond, p):
    # 9 AdaLN parameters
    c = jax.nn.silu(cond)
    c = c @ p["ada_w"] + p["ada_b"]

    s1, sc1, g1, s2, sc2, g2, s3, sc3, g3 = (
        jnp.split(c, 9, axis=-1)
    )

    # Self attention
    h = layer_norm(x, eps=1e-6)
    h = modulate(h, s1, sc1)
    x = x + g1[:, None, :] * self_attention(h, p)

    # Vision cross attention
    h = layer_norm(x, eps=1e-6)
    h = modulate(h, s2, sc2)
    vision = layer_norm(vision, eps=1e-6)
    x = x + g2[:, None, :] * cross_attention(h, vision, p)

    # MLP
    h = layer_norm(x, eps=1e-6)
    h = modulate(h, s3, sc3)

    h = jax.nn.gelu(h @ p["mlp_up"] + p["mlp_up_bias"], approximate=True)
    h = h @ p["mlp_down"] + p["mlp_down_bias"]

    return x + g3[:, None, :] * h


# ============================================================
# Full DiT
# ============================================================

def init_dit_params(key):
    k = jax.random.split(
        key,
        DIT_DEPTH + 2,
    )

    return {
        "blocks": [
            init_dit_block(k[i])
            for i in range(DIT_DEPTH)
        ],

        # Final AdaLN
        "final_ada_w": init_weight(
            k[-2],
            DIT_DIM,
            2 * DIT_DIM,
        ),
        "final_ada_b": jnp.zeros(
            2 * DIT_DIM
        ),

        # Velocity output
        "final_w": init_weight(
            k[-1],
            DIT_DIM,
            ACTION_DIM,
        ),
        "final_b": jnp.zeros(ACTION_DIM),
    }


def final_layer(x, cond, p):
    c = jax.nn.silu(cond)
    c = c @ p["final_ada_w"] + p["final_ada_b"]

    shift, scale = jnp.split(
        c, 2, axis=-1
    )

    x = layer_norm(x, eps=1e-6)

    x = modulate(
        x,
        shift,
        scale,
    )

    return x @ p["final_w"] + p["final_b"]


def dit_forward(action_tokens, vision_tokens, cond, p):
    x = action_tokens

    for block in p["blocks"]:
        x = dit_block(
            x,
            vision_tokens,
            cond,
            block,
        )

    return final_layer(
        x,
        cond,
        p,
    )


# ============================================================
# Complete ABC
# ============================================================

def init_abc_params(key):
    k = jax.random.split(key, 5)

    return {
        "vision": init_vision_params(k[0]),
        "time": init_time_params(k[1]),
        "condition": init_condition_params(k[2]),
        "action": init_action_params(k[3]),
        "dit": init_dit_params(k[4]),
    }


def abc_forward(
    dino_tokens,
    state,
    task,
    noisy_actions,
    t,
    p,
):
    # Vision
    vision = encode_vision_for_dit(
        dino_tokens,
        p["vision"],
    )

    # State + task + time
    cond = condition_embed(
        state,
        task,
        t,
        p["condition"],
        p["time"],
    )

    # Noisy action tokens
    action_tokens = action_embed(
        noisy_actions,
        p["action"],
    )

    # Predict flow velocity
    return dit_forward(
        action_tokens,
        vision,
        cond,
        p["dit"],
    )


# ============================================================
# Flow matching
# ============================================================

def flow_matching_loss(
    params,
    dino_tokens,
    state,
    task,
    actions,
    key,
):
    B = actions.shape[0]
    kt, kn = jax.random.split(key)

    # Sample time and noise
    t = jax.random.uniform(kt, (B,))
    noise = jax.random.normal(kn, actions.shape)

    # x_t = (1-t)x_0 + t*noise
    t3 = t[:, None, None]
    x_t = (1.0 - t3) * actions + t3 * noise

    # Flow target
    target = noise - actions

    pred = abc_forward(
        dino_tokens,
        state,
        task,
        x_t,
        t,
        params,
    )

    return jnp.mean(
        (pred - target) ** 2
    )


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
