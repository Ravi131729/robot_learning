import os
import pickle
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tiktoken
import wandb


# -----------------------
# Config
# -----------------------
SEED = 0
SEQUENCE_LENGTH = 64
D_MODEL = 256
HIDDEN_DIM = 2048
NUM_HEADS = 8
NUM_LAYERS = 6
BATCH_SIZE = 16              # per-device batch size
LEARNING_RATE = 1e-3
MAX_STEPS = 10_000
VAL_EVERY = 100
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "latest.pkl")

devices = jax.devices()
n_devices = len(devices)

GLOBAL_BATCH_SIZE = BATCH_SIZE * n_devices


# -----------------------
# Data
# -----------------------
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
ids = np.array(enc.encode(text), dtype=np.int32)

split_1 = int(0.9 * len(ids))
train_ids = ids[:split_1]
val_ids = ids[split_1:]


# -----------------------
# Checkpointing
# -----------------------
os.makedirs(CKPT_DIR, exist_ok=True)

def replicate_tree(tree, n_replicas):
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_replicas, axis=0), tree)

def unreplicate_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.array(jax.device_get(x[0])), tree)

def save_checkpoint(step, params, opt_state, path):
    ckpt = {
        "step": step,
        "params": unreplicate_tree(params),
        "opt_state": unreplicate_tree(opt_state),
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)

def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["step"], ckpt["params"], ckpt["opt_state"]


# -----------------------
# Batch sampling
# -----------------------
def get_batch(split: str):
    if split == "train":
        data = train_ids
    elif split == "val":
        data = val_ids
    else:
        raise ValueError(f"Unknown split: {split}")

    # valid starts: 0 .. len(data) - SEQUENCE_LENGTH - 1
    high = len(data) - SEQUENCE_LENGTH
    ix = np.random.randint(0, high, size=(GLOBAL_BATCH_SIZE,))

    x = np.stack([data[i : i + SEQUENCE_LENGTH] for i in ix])
    y = np.stack([data[i + 1 : i + SEQUENCE_LENGTH + 1] for i in ix])

    x = jnp.array(x, dtype=jnp.int32).reshape(n_devices, BATCH_SIZE, SEQUENCE_LENGTH)
    y = jnp.array(y, dtype=jnp.int32).reshape(n_devices, BATCH_SIZE, SEQUENCE_LENGTH)
    return x, y


# -----------------------
# JAX batch sampler for validation-style compute if needed
# -----------------------
val_ids_jax = jnp.array(val_ids, dtype=jnp.int32)

def get_batch_jax(data, key):
    max_start = data.shape[0] - SEQUENCE_LENGTH
    starts = jax.random.randint(key, (GLOBAL_BATCH_SIZE,), 0, max_start)

    def one(start):
        x = jax.lax.dynamic_slice(data, (start,), (SEQUENCE_LENGTH,))
        y = jax.lax.dynamic_slice(data, (start + 1,), (SEQUENCE_LENGTH,))
        return x, y

    x, y = jax.vmap(one)(starts)
    x = x.reshape(n_devices, BATCH_SIZE, SEQUENCE_LENGTH)
    y = y.reshape(n_devices, BATCH_SIZE, SEQUENCE_LENGTH)
    return x, y


# -----------------------
# Model init helpers
# -----------------------
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_hat + beta

def mlp(x, W1, b1, W2, b2):
    h = jax.nn.gelu(x @ W1 + b1)
    return h @ W2 + b2

def init_linear(key, in_dim, out_dim, scale=0.02):
    return scale * jax.random.normal(key, (in_dim, out_dim))

def init_block_params(key, d_model, hidden_dim):
    keys = jax.random.split(key, 6)
    return {
        "Wq": init_linear(keys[0], d_model, d_model),
        "Wk": init_linear(keys[1], d_model, d_model),
        "Wv": init_linear(keys[2], d_model, d_model),
        "Wo": init_linear(keys[3], d_model, d_model),
        "W1": init_linear(keys[4], d_model, hidden_dim),
        "b1": jnp.zeros((hidden_dim,)),
        "W2": init_linear(keys[5], hidden_dim, d_model),
        "b2": jnp.zeros((d_model,)),
        "gamma1": jnp.ones((d_model,)),
        "beta1": jnp.zeros((d_model,)),
        "gamma2": jnp.ones((d_model,)),
        "beta2": jnp.zeros((d_model,)),
    }

def init_gpt_params(key, vocab_size, seq_len, d_model, hidden_dim, num_layers):
    keys = jax.random.split(key, num_layers + 2)
    tok_key = keys[0]
    pos_key = keys[1]
    block_keys = keys[2:]

    return {
        "tok_emb": init_linear(tok_key, vocab_size, d_model),
        "pos_emb": init_linear(pos_key, seq_len, d_model),
        "blocks": [init_block_params(k, d_model, hidden_dim) for k in block_keys],
        "final_gamma": jnp.ones((d_model,)),
        "final_beta": jnp.zeros((d_model,)),
    }


# -----------------------
# Transformer
# -----------------------
def mha(x, p):
    B, T, D = x.shape
    H = NUM_HEADS
    assert D % H == 0
    Dh = D // H

    Q = x @ p["Wq"]
    K = x @ p["Wk"]
    V = x @ p["Wv"]

    Q = Q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / jnp.sqrt(Dh)

    mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
    scores = jnp.where(mask, scores, -1e10)

    weights = jax.nn.softmax(scores, axis=-1)
    context = weights @ V

    context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
    return context @ p["Wo"]

def transformer_block(x, p):
    x1 = layer_norm(x, p["gamma1"], p["beta1"])
    x = x + mha(x1, p)

    x2 = layer_norm(x, p["gamma2"], p["beta2"])
    x = x + mlp(x2, p["W1"], p["b1"], p["W2"], p["b2"])
    return x

def gpt_forward(params, idx):
    B, T = idx.shape
    x = params["tok_emb"][idx] + params["pos_emb"][None, :T, :]

    for blk in params["blocks"]:
        x = transformer_block(x, blk)

    x = layer_norm(x, params["final_gamma"], params["final_beta"])
    logits = x @ params["tok_emb"].T
    return logits

def loss_fn(params, input_ids, target_ids):
    logits = gpt_forward(params, input_ids)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
    return loss.mean()


# -----------------------
# pmapped train/eval
# -----------------------
tx = optax.adam(learning_rate=LEARNING_RATE)

@partial(jax.pmap, axis_name="devices")
def update(params, opt_state, input_ids, target_ids):
    loss, grads = jax.value_and_grad(loss_fn)(params, input_ids, target_ids)
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@partial(jax.pmap, axis_name="devices")
def eval_step(params, input_ids, target_ids):
    loss = loss_fn(params, input_ids, target_ids)
    loss = jax.lax.pmean(loss, axis_name="devices")
    return loss


def estimate_val_loss(params, key):
    x, y = get_batch_jax(val_ids_jax, key)
    loss = eval_step(params, x, y)
    return float(jnp.mean(loss))


# -----------------------
# Init / resume
# -----------------------
key = jax.random.PRNGKey(SEED)
key, init_key, val_key = jax.random.split(key, 3)

params = init_gpt_params(
    init_key,
    vocab_size=enc.n_vocab,
    seq_len=SEQUENCE_LENGTH,
    d_model=D_MODEL,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
)
opt_state = tx.init(params)

params = replicate_tree(params, n_devices)
opt_state = replicate_tree(opt_state, n_devices)

start_step = 0
if os.path.exists(CKPT_PATH):
    start_step, params_host, opt_state_host = load_checkpoint(CKPT_PATH)
    params = replicate_tree(_to_jax_tree(params_host), n_devices)
    opt_state = replicate_tree(_to_jax_tree(opt_state_host), n_devices)
    print(f"Loaded checkpoint from step {start_step}")


def _to_jax_tree(tree):
    return jax.tree_util.tree_map(lambda x: jnp.array(x), tree)


# -----------------------
# W&B
# -----------------------
wandb.init(
    project="jax-gpt",
    name="gpt6l_512d_pmap",
    config={
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size_per_device": BATCH_SIZE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "d_model": D_MODEL,
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
        "n_devices": n_devices,
    },
)

# -----------------------
# Training loop
# -----------------------
for step in range(start_step, MAX_STEPS):
    input_ids, target_ids = get_batch("train")
    params, opt_state, train_loss = update(params, opt_state, input_ids, target_ids)

    if step % VAL_EVERY == 0:
        val_key, subkey = jax.random.split(val_key)
        val_loss = estimate_val_loss(params, subkey)

        train_loss_scalar = float(jnp.mean(train_loss))
        print(f"Step {step} | train {train_loss_scalar:.4f} | val {val_loss:.4f}")

        wandb.log(
            {
                "train/loss": train_loss_scalar,
                "val/loss": val_loss,
            },
            step=step,
        )

        save_checkpoint(step + 1, params, opt_state, CKPT_PATH)

wandb.finish()