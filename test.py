import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


# ============================================================
# 1. GENERATE DATASET
# ============================================================

def generate_trajectory(start, goal, num_steps=20):
    """
    Generate a straight-line trajectory from start to goal.
    """

    x = np.linspace(start[0], goal[0], num_steps)
    y = np.linspace(start[1], goal[1], num_steps)

    trajectory = np.stack([x, y], axis=1)

    return trajectory


num_trajectories = 1000
num_trajectory_steps = 20

dataset = []

for _ in range(num_trajectories):

    start = np.random.uniform(0, 10, size=2)
    goal = np.random.uniform(0, 10, size=2)

    trajectory = generate_trajectory(
        start,
        goal,
        num_steps=num_trajectory_steps
    )

    dataset.append(trajectory)


dataset = np.array(dataset, dtype=np.float32)

print("Original dataset shape:", dataset.shape)


# ============================================================
# 2. NORMALIZE DATA
# ============================================================

# Original coordinates:
#
#     0 -----> 10
#
# Convert them to:
#
#    -1 -----> +1
#
# Diffusion works much better when data is roughly centered
# around zero.

dataset_normalized = dataset / 5.0 - 1.0

data = jnp.array(
    dataset_normalized,
    dtype=jnp.float32
)

print("JAX dataset shape:", data.shape)
print("Minimum:", data.min())
print("Maximum:", data.max())


# ============================================================
# 3. DIFFUSION SCHEDULE
# ============================================================

T = 1000

betas = jnp.linspace(
    0.0001,
    0.02,
    T
)

alphas = 1.0 - betas

alpha_bars = jnp.cumprod(alphas)


print("\nFinal alpha_bar:")
print(float(alpha_bars[-1]))

# This should be close to zero.
#
# That means at t=999, the trajectory is almost completely noise.


# ============================================================
# 4. SIMPLE NEURAL NETWORK
# ============================================================

trajectory_dim = num_trajectory_steps * 2

# trajectory = 40 numbers
# timestep   = 1 number

input_dim = trajectory_dim + 1

hidden_dim = 256

output_dim = trajectory_dim


def init_params(key):

    k1, k2 = jax.random.split(key)

    params = {

        "W1": (
            jax.random.normal(
                k1,
                (input_dim, hidden_dim)
            )
            * 0.05
        ),

        "b1": jnp.zeros(hidden_dim),

        "W2": (
            jax.random.normal(
                k2,
                (hidden_dim, output_dim)
            )
            * 0.05
        ),

        "b2": jnp.zeros(output_dim),
    }

    return params


# ============================================================
# 5. MODEL
# ============================================================

def model(params, xt, t):

    # --------------------------------
    # Flatten trajectory
    # --------------------------------

    x = xt.reshape(-1)

    # Shape:
    #
    # (20,2)
    #    ↓
    # (40,)


    # --------------------------------
    # Normalize timestep
    # --------------------------------

    t_normalized = t.astype(jnp.float32) / (T - 1)


    # --------------------------------
    # Add timestep to model input
    # --------------------------------

    x = jnp.concatenate(
        [
            x,
            jnp.array([t_normalized])
        ]
    )

    # Shape = (41,)


    # --------------------------------
    # Hidden layer
    # --------------------------------

    h = jax.nn.relu(
        x @ params["W1"]
        +
        params["b1"]
    )


    # --------------------------------
    # Output layer
    # --------------------------------

    output = (
        h @ params["W2"]
        +
        params["b2"]
    )


    # Return predicted noise
    # with trajectory shape

    return output.reshape(
        num_trajectory_steps,
        2
    )


# ============================================================
# 6. FORWARD DIFFUSION
# ============================================================

def add_noise(x0, t, key):

    # Generate Gaussian noise

    noise = jax.random.normal(
        key,
        shape=x0.shape
    )


    # Get alpha_bar for this timestep

    alpha_bar = alpha_bars[t]


    # DDPM forward equation:
    #
    # xt =
    #
    # sqrt(alpha_bar) * x0
    #
    #       +
    #
    # sqrt(1-alpha_bar) * noise

    xt = (
        jnp.sqrt(alpha_bar) * x0
        +
        jnp.sqrt(1.0 - alpha_bar) * noise
    )


    return xt, noise


# ============================================================
# 7. LOSS FUNCTION
# ============================================================

def loss_fn(params, x0, t, key):

    # --------------------------------
    # Add noise
    # --------------------------------

    xt, actual_noise = add_noise(
        x0,
        t,
        key
    )


    # --------------------------------
    # Ask neural network:
    #
    # "What noise was added?"
    # --------------------------------

    predicted_noise = model(
        params,
        xt,
        t
    )


    # --------------------------------
    # Compare predicted vs actual noise
    # --------------------------------

    loss = jnp.mean(
        (
            predicted_noise
            -
            actual_noise
        ) ** 2
    )


    return loss


# ============================================================
# 8. TRAINING STEP
# ============================================================

learning_rate = 1e-3


@jax.jit
def train_step(params, x0, t, key):

    loss, grads = jax.value_and_grad(
        loss_fn
    )(
        params,
        x0,
        t,
        key
    )


    # Simple SGD

    params = jax.tree.map(
        lambda p, g:
            p - learning_rate * g,
        params,
        grads
    )


    return params, loss


# ============================================================
# 9. INITIALIZE MODEL
# ============================================================

key = jax.random.PRNGKey(42)

key, init_key = jax.random.split(key)

params = init_params(init_key)


# ============================================================
# 10. TRAIN
# ============================================================

num_training_steps = 100000

loss_history = []


print("\nStarting training...\n")


for step in range(num_training_steps):

    key, data_key, t_key, noise_key = jax.random.split(
        key,
        4
    )


    # --------------------------------
    # Pick random trajectory
    # --------------------------------

    index = jax.random.randint(
        data_key,
        (),
        0,
        len(data)
    )

    x0 = data[index]


    # --------------------------------
    # Pick random diffusion timestep
    # --------------------------------

    t = jax.random.randint(
        t_key,
        (),
        0,
        T
    )


    # --------------------------------
    # Train
    # --------------------------------

    params, loss = train_step(
        params,
        x0,
        t,
        noise_key
    )


    loss_history.append(float(loss))


    if step % 1000 == 0:

        recent_loss = np.mean(
            loss_history[-500:]
        )

        print(
            f"step={step:5d} "
            f"loss={recent_loss:.4f}"
        )


print("\nTraining complete.")


# ============================================================
# 11. PLOT TRAINING LOSS
# ============================================================

# Raw loss is noisy because every iteration uses
# a different trajectory, timestep and noise sample.

window = 200

smoothed_loss = np.convolve(
    loss_history,
    np.ones(window) / window,
    mode="valid"
)


plt.figure(figsize=(8, 4))

plt.plot(smoothed_loss)

plt.xlabel("Training step")
plt.ylabel("MSE loss")

plt.title("Diffusion Training Loss")

plt.grid()

plt.show()


# ============================================================
# 12. REVERSE DIFFUSION / SAMPLING
# ============================================================

def sample(params, key):

    # --------------------------------
    # Start from pure Gaussian noise
    # --------------------------------

    key, initial_key = jax.random.split(key)

    x = jax.random.normal(
        initial_key,
        shape=(num_trajectory_steps, 2)
    )


    # --------------------------------
    # Reverse diffusion
    #
    # t = 999 -> 998 -> ... -> 0
    # --------------------------------

    for t in reversed(range(T)):

        t_array = jnp.array(
            t,
            dtype=jnp.int32
        )


        # Neural network predicts noise

        predicted_noise = model(
            params,
            x,
            t_array
        )


        alpha = alphas[t]

        alpha_bar = alpha_bars[t]

        beta = betas[t]


        # --------------------------------
        # DDPM reverse mean
        # --------------------------------

        mean = (
            1.0 / jnp.sqrt(alpha)
        ) * (
            x
            -
            (
                beta
                /
                jnp.sqrt(1.0 - alpha_bar)
            )
            *
            predicted_noise
        )


        # --------------------------------
        # Add random noise
        # except final step
        # --------------------------------

        if t > 0:

            key, noise_key = jax.random.split(key)

            noise = jax.random.normal(
                noise_key,
                shape=x.shape
            )

            x = (
                mean
                +
                jnp.sqrt(beta) * noise
            )

        else:

            x = mean


    return x


# ============================================================
# 13. GENERATE ONE TRAJECTORY
# ============================================================

key, sample_key = jax.random.split(key)

generated = sample(
    params,
    sample_key
)


# Convert JAX -> NumPy

generated_np = np.array(generated)


# ============================================================
# 14. CONVERT BACK TO ORIGINAL COORDINATES
# ============================================================

# [-1,+1] -> [0,10]

generated_original = (
    generated_np + 1.0
) * 5.0


# ============================================================
# 15. VISUALIZE GENERATED TRAJECTORY
# ============================================================

plt.figure(figsize=(7, 7))


# Plot some real training trajectories

for i in range(30):

    trajectory = dataset[i]

    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        alpha=0.12
    )


# Plot generated trajectory

plt.plot(
    generated_original[:, 0],
    generated_original[:, 1],
    "o-",
    linewidth=3,
    label="Generated trajectory"
)


# Mark generated start

plt.scatter(
    generated_original[0, 0],
    generated_original[0, 1],
    s=120,
    marker="o",
    label="Generated start"
)


# Mark generated end

plt.scatter(
    generated_original[-1, 0],
    generated_original[-1, 1],
    s=120,
    marker="x",
    label="Generated end"
)


plt.xlim(-1, 11)
plt.ylim(-1, 11)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "Real Training Trajectories vs Diffusion Sample"
)

plt.grid()
plt.legend()

plt.show()


# ============================================================
# 16. GENERATE MULTIPLE SAMPLES
# ============================================================

plt.figure(figsize=(7, 7))


for i in range(10):

    key, sample_key = jax.random.split(key)

    generated = sample(
        params,
        sample_key
    )

    generated = np.array(generated)

    # Convert back to [0,10]

    generated = (
        generated + 1.0
    ) * 5.0


    plt.plot(
        generated[:, 0],
        generated[:, 1],
        "o-",
        alpha=0.7
    )


plt.xlim(-1, 11)
plt.ylim(-1, 11)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "10 Trajectories Generated by Diffusion"
)

plt.grid()

plt.show()