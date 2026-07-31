import jax
import jax.numpy as jnp



key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 15)

keyX, keyWq1, keyWk1, keyWv1, keyWq2, keyWk2, keyWv2, keyWo, keyW1, keyW2 = keys[:10]


T = 50 # sequence length
num_heads = 8
d_model =512
hidden_dim = 128
# d_head = d_model // num_heads

X = jax.random.normal(keyX, (T, d_model))

W_q = jax.random.normal(keyWq1, (d_model, d_model))
W_k = jax.random.normal(keyWk1, (d_model, d_model))
W_v = jax.random.normal(keyWv1, (d_model, d_model))



Wo = jax.random.normal(keyWo, (d_model, d_model))   # back to model dim

W1 = jax.random.normal(keyW1, (d_model, hidden_dim))
b1 = jnp.zeros((hidden_dim,))

W2 = jax.random.normal(keyW2, (hidden_dim, d_model))
b2 = jnp.zeros((d_model,))

gamma = jnp.ones((d_model,))
beta = jnp.zeros((d_model,))
gamma2 = jnp.ones((d_model,))
beta2 = jnp.zeros((d_model,))

# def attention(X, Wq, Wk, Wv):
#     Q = X @ Wq
#     K = X @ Wk
#     V = X @ Wv
#     scores = (Q @ K.T) / jnp.sqrt(Q.shape[-1])
#     weights = jax.nn.softmax(scores, axis=-1)
#     return weights @ V

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_hat + beta

def mlp(x, W1, b1, W2, b2):
    h = jax.nn.gelu(x @ W1 + b1)
    return h @ W2 + b2

def mha(x, wq, wk, wv, wo, num_heads):
    T, d_model = x.shape
    assert d_model % num_heads == 0
    d_head = d_model // num_heads

    Q = x @ wq
    K = x @ wk
    V = x @ wv

    Q = Q.reshape(T, num_heads, d_head).transpose(1, 0, 2)  # (H, T, Dh)
    K = K.reshape(T, num_heads, d_head).transpose(1, 0, 2)
    V = V.reshape(T, num_heads, d_head).transpose(1, 0, 2)

    mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, :, :]  # (1, T, T)

    scores = (Q @ K.transpose(0, 2, 1)) / jnp.sqrt(d_head)      # (H, T, T)
    scores = jnp.where(mask, scores, -1e10)

    weights = jax.nn.softmax(scores, axis=-1)
    context = weights @ V                                       # (H, T, Dh)

    context = context.transpose(1, 0, 2).reshape(T, d_model)    # (T, D)
    return context @ wo

x1 = layer_norm(X, gamma, beta)
attn_out = mha(x1, W_q, W_k, W_v, Wo, num_heads=2)
x2 = X + attn_out
x3 = layer_norm(x2, gamma2, beta2)
mlp_out = mlp(x3, W1, b1, W2, b2)
output = x2 + mlp_out

print("X", X.shape)
print("attn_out", attn_out.shape)
print("x2", x2.shape)
print("x3", x3.shape)
print("mlp_out", mlp_out.shape)
print("output", output.shape)