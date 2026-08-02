import os
import pickle
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
D_MODEL = 512
HIDDEN_DIM = 2048   # more standard than 128 for D_MODEL=512
NUM_HEADS = 8
NUM_LAYERS = 6
BATCH_SIZE =128
LEARNING_RATE = 1e-3
MAX_STEPS = 10_000
VAL_EVERY = 100
VAL_BATCHES = 10
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "latest.pkl")


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

def _to_device_tree(tree):
    return jax.tree_util.tree_map(lambda x: jnp.array(x), tree)

def save_checkpoint(step, params, opt_state, path):
    ckpt = {
        "step": step,
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
    }
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)

def load_checkpoint(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    step = ckpt["step"]
    params = _to_device_tree(ckpt["params"])
    opt_state = _to_device_tree(ckpt["opt_state"])
    return step, params, opt_state


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

    max_start = len(data) - SEQUENCE_LENGTH - 1
    ix = np.random.randint(0, max_start, size=(BATCH_SIZE,))

    x = np.stack([data[i : i + SEQUENCE_LENGTH] for i in ix])
    y = np.stack([data[i + 1 : i + SEQUENCE_LENGTH + 1] for i in ix])

    return jnp.array(x, dtype=jnp.int32), jnp.array(y, dtype=jnp.int32)

#-----------------------
# jax version of get_batch
#-----------------------
val_ids_jax = jnp.array(val_ids, dtype=jnp.int32)

def get_batch_jax(data, key):
    max_start = data.shape[0] - SEQUENCE_LENGTH - 1
    starts = jax.random.randint(key, (BATCH_SIZE,), 0, max_start)

    def one(start):
        x = jax.lax.dynamic_slice(data, (start,), (SEQUENCE_LENGTH,))
        y = jax.lax.dynamic_slice(data, (start + 1,), (SEQUENCE_LENGTH,))
        return x, y

    return jax.vmap(one)(starts)
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
        "tok_emb": init_linear(tok_key, vocab_size, d_model),   # (V, D)
        "pos_emb": init_linear(pos_key, seq_len, d_model),      # (T, D)
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

    Q = x @ p["Wq"]  # (B, T, D)
    K = x @ p["Wk"]
    V = x @ p["Wv"]

    Q = Q.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)  # (B, H, T, Dh)
    K = K.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / jnp.sqrt(Dh)  # (B, H, T, T)

    mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
    scores = jnp.where(mask, scores, -1e10)

    weights = jax.nn.softmax(scores, axis=-1)
    context = weights @ V  # (B, H, T, Dh)

    context = context.transpose(0, 2, 1, 3).reshape(B, T, D)  # (B, T, D)
    return context @ p["Wo"]  # important: output projection

def transformer_block(x, p):
    x1 = layer_norm(x, p["gamma1"], p["beta1"])
    x = x + mha(x1, p)

    x2 = layer_norm(x, p["gamma2"], p["beta2"])
    x = x + mlp(x2, p["W1"], p["b1"], p["W2"], p["b2"])
    return x

def gpt_forward(params, idx):
    """
    idx: (B, T) int32 token ids
    returns logits: (B, T, V)
    """
    B, T = idx.shape
    x = params["tok_emb"][idx] + params["pos_emb"][None, :T, :]  # (B, T, D)

    for blk in params["blocks"]:
        x = transformer_block(x, blk)

    x = layer_norm(x, params["final_gamma"], params["final_beta"])
    logits = x @ params["tok_emb"].T  # tied output head
    return logits

def loss_fn(params, input_ids, target_ids):
    logits = gpt_forward(params, input_ids)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_ids)
    return loss.mean()


# -----------------------
# Training / eval steps
# -----------------------
key = jax.random.PRNGKey(SEED)
key,val_key = jax.random.split(key)
params = init_gpt_params(
    key,
    vocab_size=enc.n_vocab,
    seq_len=SEQUENCE_LENGTH,
    d_model=D_MODEL,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
)

tx = optax.adam(learning_rate=LEARNING_RATE)
opt_state = tx.init(params)

@jax.jit
def update(params, opt_state, input_ids, target_ids):
    loss, grads = jax.value_and_grad(loss_fn)(params, input_ids, target_ids)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def eval_loss(params, input_ids, target_ids):
    return loss_fn(params, input_ids, target_ids)

@jax.jit
def estimate_val_loss(params, key):

    keys = jax.random.split(key, VAL_BATCHES)

    def body(total_loss, key):
        x, y = get_batch_jax(val_ids_jax, key)
        loss = loss_fn(params, x, y)
        return total_loss + loss, None

    total_loss, _ = jax.lax.scan(
        body,
        0.0,
        keys,
    )

    return total_loss / VAL_BATCHES


wandb.init(
    project="jax-gpt",
    name="gpt6l_512d",
    config={
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "d_model": D_MODEL,
        "hidden_dim": HIDDEN_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
    },
)
# -----------------------
# Resume if checkpoint exists
# -----------------------
start_step = 0
if os.path.exists(CKPT_PATH):
    start_step, params, opt_state = load_checkpoint(CKPT_PATH)
    print(f"Loaded checkpoint from step {start_step}")


# -----------------------
# Training loop
# -----------------------

for step in range(start_step, MAX_STEPS):
    input_ids, target_ids = get_batch("train")
    params, opt_state, train_loss = update(params, opt_state, input_ids, target_ids)

    if step % VAL_EVERY == 0:
        val_key, subkey = jax.random.split(val_key)
        val_loss = estimate_val_loss(params, subkey)

        print(
            f"Step {step} | "
            f"train {float(train_loss):.4f} | "
            f"val {float(val_loss):.4f}"
        )

        wandb.log(
            {
                "train/loss": float(train_loss),
                "val/loss": float(val_loss),
            },
            step=step,
        )

        save_checkpoint(step + 1, params, opt_state, CKPT_PATH)

wandb.finish()