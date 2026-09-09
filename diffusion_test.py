import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from flax import nnx
import optax


# ============================================================
# 1. CONFIG
# ============================================================

H = 32

T = 1000

NUM_TRAJECTORIES = 8000
NUM_TRAIN_STEPS = 20000

TIME_DIM = 64
COND_DIM = 3

HIDDEN_DIM = 256

LEARNING_RATE = 1e-3


# ============================================================
# 2. FIXED Y COORDINATES
# ============================================================

y_coordinates = jnp.linspace(
    0.0,
    1.0,
    H
)


# ============================================================
# 3. CREATE CONDITIONAL TRAJECTORY
#
# condition =
#
# [
#     obstacle_x,
#     obstacle_y,
#     obstacle_radius
# ]
#
# ============================================================

def make_trajectory(key):

    (
        key_obs_x,
        key_obs_y,
        key_radius,
        key_side,
        key_clearance,
        key_skew
    ) = jax.random.split(
        key,
        6
    )


    # --------------------------------------------------------
    # Random obstacle
    # --------------------------------------------------------

    obstacle_x = jax.random.uniform(
        key_obs_x,
        minval=-0.15,
        maxval=0.15
    )


    obstacle_y = jax.random.uniform(
        key_obs_y,
        minval=0.35,
        maxval=0.65
    )


    radius = jax.random.uniform(
        key_radius,
        minval=0.10,
        maxval=0.18
    )


    # --------------------------------------------------------
    # Choose which side of obstacle
    #
    # -1 = left
    # +1 = right
    # --------------------------------------------------------

    side = jax.random.choice(
        key_side,
        jnp.array([-1.0, 1.0])
    )


    # --------------------------------------------------------
    # Random additional clearance
    # --------------------------------------------------------

    clearance = jax.random.uniform(
        key_clearance,
        minval=0.08,
        maxval=0.20
    )


    # --------------------------------------------------------
    # Desired x position around obstacle
    # --------------------------------------------------------

    target_x = (
        obstacle_x
        +
        side
        * (
            radius
            + clearance
        )
    )


    # ========================================================
    # Create smooth path
    #
    # We want:
    #
    # x(0) = 0
    # x(1) = 0
    #
    # and maximum deviation near obstacle_y.
    # ========================================================


    # --------------------------------------------------------
    # Gaussian centered around obstacle
    # --------------------------------------------------------

    width = 0.20


    bump = jnp.exp(
        -0.5
        *
        (
            (y_coordinates - obstacle_y)
            / width
        ) ** 2
    )


    # --------------------------------------------------------
    # Envelope forces endpoints to zero
    # --------------------------------------------------------

    envelope = jnp.sin(
        jnp.pi
        * y_coordinates
    )


    shape = (
        bump
        * envelope
    )


    # --------------------------------------------------------
    # Normalize so trajectory reaches approximately target_x
    # --------------------------------------------------------

    shape = (
        shape
        /
        jnp.max(shape)
    )


    x = (
        target_x
        * shape
    )


    # --------------------------------------------------------
    # Add smooth asymmetry
    # --------------------------------------------------------

    skew_strength = jax.random.uniform(
        key_skew,
        minval=-0.06,
        maxval=0.06
    )


    x = (
        x
        +
        skew_strength
        * jnp.sin(
            2.0
            * jnp.pi
            * y_coordinates
        )
    )


    # --------------------------------------------------------
    # Hard endpoints
    # --------------------------------------------------------

    x = x.at[0].set(0.0)

    x = x.at[-1].set(0.0)


    # --------------------------------------------------------
    # Condition
    # --------------------------------------------------------

    condition = jnp.array(
        [
            obstacle_x,
            obstacle_y,
            radius
        ]
    )


    return x, condition


# ============================================================
# 4. GENERATE DATASET
# ============================================================

key = jax.random.PRNGKey(0)


keys = jax.random.split(
    key,
    NUM_TRAJECTORIES
)


x_data, conditions = jax.vmap(
    make_trajectory
)(keys)


print(
    "trajectory shape:",
    x_data.shape
)

print(
    "condition shape:",
    conditions.shape
)


# ============================================================
# 5. VISUALIZE RANDOM TRAINING EXAMPLES
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 7)
)


for i, ax in enumerate(
    axes.flatten()
):

    x = x_data[i]

    obstacle_x = conditions[i, 0]
    obstacle_y = conditions[i, 1]
    radius = conditions[i, 2]


    ax.plot(
        x,
        y_coordinates,
        linewidth=2
    )


    circle = plt.Circle(
        (
            float(obstacle_x),
            float(obstacle_y)
        ),
        float(radius),
        color="blue",
        alpha=0.5
    )


    ax.add_artist(
        circle
    )


    ax.scatter(
        0,
        0
    )

    ax.scatter(
        0,
        1
    )


    ax.set_xlim(
        -0.7,
        0.7
    )

    ax.set_ylim(
        -0.1,
        1.1
    )

    ax.set_aspect(
        "equal"
    )


plt.tight_layout()

plt.show()


# ============================================================
# 6. DIFFUSION SCHEDULE
# ============================================================

betas = jnp.linspace(
    1e-4,
    0.02,
    T
)


alphas = (
    1.0
    -
    betas
)


alpha_bars = jnp.cumprod(
    alphas
)


print(
    "alpha_bar_T:",
    alpha_bars[-1]
)


# ============================================================
# 7. ENDPOINT MASK
#
# Do not diffuse:
#
# x[0]
# x[-1]
# ============================================================

endpoint_mask = jnp.ones(
    H
)


endpoint_mask = endpoint_mask.at[
    0
].set(
    0.0
)


endpoint_mask = endpoint_mask.at[
    -1
].set(
    0.0
)


# ============================================================
# 8. FORWARD DIFFUSION
# ============================================================

def make_noisy_batch(
    key,
    x0
):

    batch_size = x0.shape[0]


    key_t, key_noise = jax.random.split(
        key
    )


    # --------------------------------------------------------
    # Random timestep
    # --------------------------------------------------------

    t = jax.random.randint(
        key_t,
        shape=(batch_size,),
        minval=1,
        maxval=T + 1
    )


    # --------------------------------------------------------
    # Gaussian noise
    # --------------------------------------------------------

    noise = jax.random.normal(
        key_noise,
        shape=x0.shape
    )


    # --------------------------------------------------------
    # No noise at endpoints
    # --------------------------------------------------------

    noise = (
        noise
        * endpoint_mask
    )


    # --------------------------------------------------------
    # alpha_bar_t
    # --------------------------------------------------------

    alpha_bar_t = alpha_bars[
        t - 1
    ]


    alpha_bar_t = alpha_bar_t[
        :, None
    ]


    # --------------------------------------------------------
    # q(x_t | x_0)
    # --------------------------------------------------------

    xt = (

        jnp.sqrt(
            alpha_bar_t
        )

        * x0

        +

        jnp.sqrt(
            1.0
            -
            alpha_bar_t
        )

        * noise
    )


    # --------------------------------------------------------
    # Hard endpoint condition
    # --------------------------------------------------------

    xt = xt.at[
        :, 0
    ].set(
        0.0
    )


    xt = xt.at[
        :, -1
    ].set(
        0.0
    )


    return (
        xt,
        t,
        noise
    )


# ============================================================
# 9. TIME EMBEDDING
# ============================================================

def timestep_embedding(
    t,
    dim=TIME_DIM
):

    half = (
        dim // 2
    )


    frequencies = jnp.exp(

        -jnp.log(
            10000.0
        )

        * jnp.arange(
            half
        )

        / (
            half - 1
        )
    )


    # Normalize timestep
    t = (
        t
        /
        T
    )


    args = (

        t[:, None]

        * frequencies[
            None,
            :
        ]

        * 1000.0
    )


    embedding = jnp.concatenate(
        [
            jnp.sin(
                args
            ),

            jnp.cos(
                args
            )
        ],
        axis=-1
    )


    return embedding


# ============================================================
# 10. CONDITIONAL DIFFUSION MODEL
#
#
# epsilon_theta(
#
#       x_t,
#       t,
#       obstacle
#
# )
#
#
# Inputs:
#
# x_t:
#
#       32
#
# timestep:
#
#       64
#
# condition:
#
#       obstacle_x
#       obstacle_y
#       radius
#
#       = 3
#
#
# total = 99
# ============================================================

class DiffusionPlanner(
    nnx.Module
):

    def __init__(
        self,
        rngs: nnx.Rngs
    ):


        self.linear1 = nnx.Linear(

            H
            +
            TIME_DIM
            +
            COND_DIM,

            HIDDEN_DIM,

            rngs=rngs
        )


        self.linear2 = nnx.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM,
            rngs=rngs
        )


        self.linear3 = nnx.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM,
            rngs=rngs
        )


        self.linear4 = nnx.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM,
            rngs=rngs
        )


        self.output = nnx.Linear(
            HIDDEN_DIM,
            H,
            rngs=rngs
        )


    def __call__(
        self,
        x,
        t,
        condition
    ):


        # ----------------------------------------------------
        # Time embedding
        # ----------------------------------------------------

        t_emb = timestep_embedding(
            t
        )


        # ----------------------------------------------------
        # Combine:
        #
        # noisy trajectory
        # +
        # timestep
        # +
        # obstacle
        # ----------------------------------------------------

        h = jnp.concatenate(
            [
                x,
                t_emb,
                condition
            ],
            axis=-1
        )


        # ----------------------------------------------------
        # MLP
        # ----------------------------------------------------

        h = self.linear1(
            h
        )

        h = nnx.silu(
            h
        )


        h = self.linear2(
            h
        )

        h = nnx.silu(
            h
        )


        h = self.linear3(
            h
        )

        h = nnx.silu(
            h
        )


        h = self.linear4(
            h
        )

        h = nnx.silu(
            h
        )


        predicted_noise = self.output(
            h
        )


        # ----------------------------------------------------
        # endpoints have zero noise
        # ----------------------------------------------------

        predicted_noise = (

            predicted_noise

            * endpoint_mask
        )


        return predicted_noise


# ============================================================
# 11. CREATE MODEL
# ============================================================


model = DiffusionPlanner(
    rngs=nnx.Rngs(0)
)


# ============================================================
# 12. LOSS
# ============================================================

def loss_fn(
    model,
    xt,
    t,
    condition,
    noise
):


    predicted_noise = model(
        xt,
        t,
        condition
    )


    error = (
        predicted_noise
        -
        noise
    )


    loss = jnp.sum(
        error ** 2
    ) / (

        error.shape[0]

        * (
            H - 2
        )
    )


    return loss


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = nnx.Optimizer(

    model,

    optax.adam(
        LEARNING_RATE
    ),

    wrt=nnx.Param
)


# ============================================================
# 14. TRAIN STEP
# ============================================================

@nnx.jit
def train_step(
    model,
    optimizer,
    xt,
    t,
    condition,
    noise
):


    loss, grads = nnx.value_and_grad(
        loss_fn
    )(
        model,
        xt,
        t,
        condition,
        noise
    )


    optimizer.update(
        model,
        grads
    )


    return loss


# ============================================================
# 15. TRAIN
# ============================================================

key = jax.random.PRNGKey(
    42
)


losses = []


for step in range(
    NUM_TRAIN_STEPS
):


    key, subkey = jax.random.split(
        key
    )


    # --------------------------------------------------------
    # Forward diffusion
    # --------------------------------------------------------

    xt, t, noise = make_noisy_batch(
        subkey,
        x_data
    )


    # --------------------------------------------------------
    # Learn epsilon
    # --------------------------------------------------------

    loss = train_step(
        model,
        optimizer,
        xt,
        t,
        conditions,
        noise
    )


    losses.append(
        float(loss)
    )


    if step % 500 == 0:

        print(
            f"step {step:6d}"
            f" | loss {loss:.6f}"
        )


# ============================================================
# 16. LOSS CURVE
# ============================================================

plt.figure(
    figsize=(7, 4)
)


plt.plot(
    losses
)


plt.xlabel(
    "training step"
)

plt.ylabel(
    "noise MSE"
)

plt.title(
    "Conditional diffusion planner"
)


plt.show()


# ============================================================
# 17. DDPM REVERSE STEP
# ============================================================

def reverse_step(
    key,
    model,
    xt,
    t,
    condition
):


    index = (
        t - 1
    )


    beta_t = betas[
        index
    ]


    alpha_t = alphas[
        index
    ]


    alpha_bar_t = alpha_bars[
        index
    ]


    batch_size = xt.shape[0]


    t_batch = jnp.full(
        (
            batch_size,
        ),
        t
    )


    # --------------------------------------------------------
    # CONDITIONAL noise prediction
    # --------------------------------------------------------

    predicted_noise = model(
        xt,
        t_batch,
        condition
    )


    # --------------------------------------------------------
    # DDPM reverse mean
    # --------------------------------------------------------

    mean = (

        1.0
        /
        jnp.sqrt(
            alpha_t
        )

    ) * (

        xt

        -

        (

            beta_t

            /

            jnp.sqrt(
                1.0
                -
                alpha_bar_t
            )

        )

        * predicted_noise
    )


    # --------------------------------------------------------
    # Endpoint conditioning
    # --------------------------------------------------------

    mean = mean.at[
        :, 0
    ].set(
        0.0
    )


    mean = mean.at[
        :, -1
    ].set(
        0.0
    )


    # --------------------------------------------------------
    # Last step
    # --------------------------------------------------------

    if t == 1:

        return mean


    # --------------------------------------------------------
    # Posterior variance
    # --------------------------------------------------------

    alpha_bar_prev = alpha_bars[
        index - 1
    ]


    beta_tilde = (

        beta_t

        *

        (
            1.0
            -
            alpha_bar_prev
        )

        /

        (
            1.0
            -
            alpha_bar_t
        )
    )


    # --------------------------------------------------------
    # Random reverse noise
    # --------------------------------------------------------

    z = jax.random.normal(
        key,
        shape=xt.shape
    )


    z = (
        z
        * endpoint_mask
    )


    xt_prev = (

        mean

        +

        jnp.sqrt(
            beta_tilde
        )

        * z
    )


    # --------------------------------------------------------
    # Endpoint conditioning
    # --------------------------------------------------------

    xt_prev = xt_prev.at[
        :, 0
    ].set(
        0.0
    )


    xt_prev = xt_prev.at[
        :, -1
    ].set(
        0.0
    )


    return xt_prev


# ============================================================
# 18. PLANNER
#
# Give obstacle:
#
# (x, y, radius)
#
# Get many possible trajectories.
# ============================================================

def plan(
    key,
    model,
    obstacle_x,
    obstacle_y,
    radius,
    num_samples=100
):


    # --------------------------------------------------------
    # Planning condition
    # --------------------------------------------------------

    condition = jnp.array(
        [
            obstacle_x,
            obstacle_y,
            radius
        ]
    )


    # --------------------------------------------------------
    # Same condition for every candidate trajectory
    # --------------------------------------------------------

    condition = jnp.broadcast_to(
        condition,
        (
            num_samples,
            COND_DIM
        )
    )


    # --------------------------------------------------------
    # Start from Gaussian noise
    # --------------------------------------------------------

    key, noise_key = jax.random.split(
        key
    )


    x = jax.random.normal(
        noise_key,
        shape=(
            num_samples,
            H
        )
    )


    # --------------------------------------------------------
    # Fixed start / goal
    # --------------------------------------------------------

    x = x.at[
        :, 0
    ].set(
        0.0
    )


    x = x.at[
        :, -1
    ].set(
        0.0
    )


    # --------------------------------------------------------
    # Reverse diffusion
    # --------------------------------------------------------

    for t in range(
        T,
        0,
        -1
    ):


        key, subkey = jax.random.split(
            key
        )


        x = reverse_step(
            subkey,
            model,
            x,
            t,
            condition
        )


    return x


# ============================================================
# 19. TEST NEW OBSTACLE
#
# This obstacle is supplied at inference time.
# ============================================================

test_obstacle_x = 0.3
test_obstacle_y = 0.55
test_radius = 0.15


key = jax.random.PRNGKey(
    999
)


planned_x = plan(
    key,
    model,
    obstacle_x=test_obstacle_x,
    obstacle_y=test_obstacle_y,
    radius=test_radius,
    num_samples=200
)


print(
    planned_x.shape
)


# ============================================================
# 20. PLOT PLANS
# ============================================================

plt.figure(
    figsize=(7, 7)
)


for trajectory_x in planned_x:

    plt.plot(
        trajectory_x,
        y_coordinates,
        alpha=0.08
    )


circle = plt.Circle(
    (
        test_obstacle_x,
        test_obstacle_y
    ),
    test_radius,
    color="blue"
)


plt.gca().add_artist(
    circle
)


plt.scatter(
    0,
    0,
    s=80,
    label="Start"
)


plt.scatter(
    0,
    1,
    s=80,
    label="Goal"
)


plt.xlim(
    -0.8,
    0.8
)


plt.ylim(
    -0.1,
    1.1
)


plt.axis(
    "equal"
)


plt.legend()


plt.title(
    "Conditional diffusion planner"
)


plt.show()