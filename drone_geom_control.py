import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SO(3) utilities
# ============================================================

def hat(w):
    """R^3 -> so(3)."""
    return np.array([
        [0.0,   -w[2],  w[1]],
        [w[2],   0.0,  -w[0]],
        [-w[1],  w[0],  0.0]
    ])


def vee(S):
    """so(3) -> R^3."""
    return np.array([
        S[2, 1],
        S[0, 2],
        S[1, 0]
    ])


def project_so3(R):
    """Numerically project a matrix back onto SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt

    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt

    return Rn


# ============================================================
# Standard quadrotor parameters
# ============================================================

m = 2.5                  # kg
g = 9.81                 # m/s^2

J = np.diag([
    0.105,
    0.105,
    0.140
])

J_inv = np.linalg.inv(J)

e3 = np.array([0.0, 0.0, 1.0])


# ============================================================
# Controller gains
# ============================================================

# Position controller
Kx = np.diag([6.0, 6.0, 8.0])
Kv = np.diag([4.5, 4.5, 5.0])

# Attitude controller
KR = np.diag([8.0, 8.0, 3.0])
KOmega = np.diag([0.25, 0.25, 0.15])


# ============================================================
# Reference trajectory
# ============================================================

# def reference(t):
#     """
#     Sine-wave reference trajectory in the X-Z plane.

#     x: moves forward at constant velocity
#     y: stays at zero
#     z: sinusoidal motion
#     """

#     # Trajectory parameters
#     vx = 0.5       # forward speed [m/s]
#     z0 = 2.0       # mean altitude [m]
#     A = 1.0        # sine amplitude [m]
#     omega = 0.8    # angular frequency [rad/s]

#     # --------------------------------
#     # Position
#     # --------------------------------
#     x = vx * t
#     y = 0.0
#     z = z0 + A * np.sin(omega * t)

#     xd = np.array([x, y, z])

#     # --------------------------------
#     # Velocity
#     # --------------------------------
#     vx_d = A * omega * np.cos(omega * t)
#     vy_d = 0.0
#     vz_d = A * omega * np.cos(omega * t)

#     vd = np.array([
#         vx_d,
#         vy_d,
#         vz_d
#     ])

#     # --------------------------------
#     # Acceleration
#     # --------------------------------
#     ax_d = 0.0
#     ay_d = 0.0
#     az_d = -A * omega**2 * np.sin(omega * t)

#     ad = np.array([
#         ax_d,
#         ay_d,
#         az_d
#     ])

#     # Fixed yaw
#     yaw = 0.0

#     return xd, vd, ad, yaw

def reference(t):
    """
    Upward spiral (helical) reference trajectory.

    x-y: circular motion
    z:   increases at constant velocity
    """

    # --------------------------------
    # Trajectory parameters
    # --------------------------------
    R = 2.0         # spiral radius [m]
    omega = 0.8     # angular speed [rad/s]
    z0 = 1.0        # initial altitude [m]
    vz = 0.3        # upward speed [m/s]

    # --------------------------------
    # Position
    # --------------------------------
    x = R * np.cos(omega * t)
    y = R * np.sin(omega * t)
    z = z0 + vz * t

    xd = np.array([
        x,
        y,
        z
    ])

    # --------------------------------
    # Velocity
    # --------------------------------
    vx_d = -R * omega * np.sin(omega * t)
    vy_d =  R * omega * np.cos(omega * t)
    vz_d = vz

    vd = np.array([
        vx_d,
        vy_d,
        vz_d
    ])

    # --------------------------------
    # Acceleration
    # --------------------------------
    ax_d = -R * omega**2 * np.cos(omega * t)
    ay_d = -R * omega**2 * np.sin(omega * t)
    az_d = 0.0

    ad = np.array([
        ax_d,
        ay_d,
        az_d
    ])

    # --------------------------------
    # Fixed yaw
    # --------------------------------
    yaw = 0.0

    return xd, vd, ad, yaw
# ============================================================
# Desired attitude from desired force
# ============================================================

def desired_rotation(Fd, yaw):
    """
    Construct desired rotation matrix.

    b3d aligns with desired total force.
    yaw specifies desired heading.
    """

    norm_F = np.linalg.norm(Fd)

    if norm_F < 1e-8:
        b3d = e3.copy()
    else:
        b3d = Fd / norm_F

    # Desired heading direction
    b1c = np.array([
        np.cos(yaw),
        np.sin(yaw),
        0.0
    ])

    b2d = np.cross(b3d, b1c)

    if np.linalg.norm(b2d) < 1e-8:
        # Degenerate case
        b1c = np.array([0.0, 1.0, 0.0])
        b2d = np.cross(b3d, b1c)

    b2d /= np.linalg.norm(b2d)

    b1d = np.cross(b2d, b3d)
    b1d /= np.linalg.norm(b1d)

    Rd = np.column_stack((b1d, b2d, b3d))

    return Rd


# ============================================================
# Geometric controller
# ============================================================

def geometric_controller(x, v, R, Omega, t):

    xd, vd, ad, yaw_d = reference(t)

    # --------------------------------------------------------
    # Position / velocity errors
    # --------------------------------------------------------

    ex = x - xd
    ev = v - vd

    # --------------------------------------------------------
    # Desired total force
    #
    # Dynamics convention:
    #
    #     m*x_ddot = -m*g*e3 + f*R*e3
    #
    # --------------------------------------------------------

    Fd = (
        -Kx @ ex
        -Kv @ ev
        +m * g * e3
        +m * ad
    )

    # --------------------------------------------------------
    # Desired attitude
    # --------------------------------------------------------

    Rd = desired_rotation(Fd, yaw_d)

    # --------------------------------------------------------
    # Geometric attitude error
    # --------------------------------------------------------

    eR_matrix = 0.5 * (
        Rd.T @ R
        -
        R.T @ Rd
    )

    eR = vee(eR_matrix)

    # For this simple trajectory we neglect Omega_d feedforward.
    # This is commonly sufficient for a basic implementation.
    Omega_d = np.zeros(3)

    eOmega = Omega - R.T @ Rd @ Omega_d

    # --------------------------------------------------------
    # Thrust
    # --------------------------------------------------------

    f = Fd @ (R @ e3)

    # Prevent negative thrust
    f = max(0.0, f)

    # --------------------------------------------------------
    # Moment
    # --------------------------------------------------------

    M = (
        -KR @ eR
        -KOmega @ eOmega
        +np.cross(Omega, J @ Omega)
    )

    return f, M, xd, vd, Rd, ex


# ============================================================
# Quadrotor rigid-body dynamics
# ============================================================

def dynamics(x, v, R, Omega, f, M):

    # Translational dynamics
    x_dot = v

    v_dot = (
        -g * e3
        +(f / m) * (R @ e3)
    )

    # Rotational kinematics
    R_dot = R @ hat(Omega)

    # Euler rigid-body rotational dynamics
    Omega_dot = J_inv @ (
        M
        -np.cross(Omega, J @ Omega)
    )

    return x_dot, v_dot, R_dot, Omega_dot


# ============================================================
# Simulation
# ============================================================

dt = 0.001
T = 20.0

N = int(T / dt)

time = np.arange(N) * dt


# ============================================================
# Initial state
# ============================================================

x = np.array([
    0.0,
    0.0,
    2.0
])

v = np.zeros(3)

R = np.eye(3)

Omega = np.zeros(3)


# ============================================================
# Data storage
# ============================================================

position_history = np.zeros((N, 3))
reference_history = np.zeros((N, 3))
velocity_history = np.zeros((N, 3))
error_history = np.zeros((N, 3))

thrust_history = np.zeros(N)
moment_history = np.zeros((N, 3))


# ============================================================
# Main simulation loop
# ============================================================

for k, t in enumerate(time):

    # --------------------------------------------------------
    # Controller
    # --------------------------------------------------------

    f, M, xd, vd, Rd, ex = geometric_controller(
        x,
        v,
        R,
        Omega,
        t
    )

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------

    position_history[k] = x
    reference_history[k] = xd
    velocity_history[k] = v
    error_history[k] = ex

    thrust_history[k] = f
    moment_history[k] = M

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------

    x_dot, v_dot, R_dot, Omega_dot = dynamics(
        x,
        v,
        R,
        Omega,
        f,
        M
    )

    # --------------------------------------------------------
    # Euler integration
    # --------------------------------------------------------

    x = x + dt * x_dot
    v = v + dt * v_dot
    R = R + dt * R_dot
    Omega = Omega + dt * Omega_dot

    # Numerical correction to keep R on SO(3)
    R = project_so3(R)


# ============================================================
# Tracking performance
# ============================================================

error_norm = np.linalg.norm(error_history, axis=1)

rmse = np.sqrt(
    np.mean(error_history**2, axis=0)
)

total_rmse = np.sqrt(
    np.mean(error_norm**2)
)

print("\nTracking RMSE")
print("---------------------------")
print(f"X RMSE : {rmse[0]:.4f} m")
print(f"Y RMSE : {rmse[1]:.4f} m")
print(f"Z RMSE : {rmse[2]:.4f} m")
print(f"Total  : {total_rmse:.4f} m")


# ============================================================
# 3-D trajectory plot
# ============================================================

fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    reference_history[:, 0],
    reference_history[:, 1],
    reference_history[:, 2],
    "--",
    linewidth=2,
    label="Reference"
)

ax.plot(
    position_history[:, 0],
    position_history[:, 1],
    position_history[:, 2],
    linewidth=2,
    label="Quadrotor"
)

ax.scatter(
    position_history[0, 0],
    position_history[0, 1],
    position_history[0, 2],
    marker="o",
    s=60,
    label="Start"
)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

ax.set_title(
    "SE(3) Geometric Control - 3D Trajectory Tracking"
)

ax.legend()
ax.grid(True)

plt.tight_layout()


# ============================================================
# XYZ tracking
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
    sharex=True
)

labels = ["X [m]", "Y [m]", "Z [m]"]

for i in range(3):

    axes[i].plot(
        time,
        reference_history[:, i],
        "--",
        label="Reference"
    )

    axes[i].plot(
        time,
        position_history[:, i],
        label="Actual"
    )

    axes[i].set_ylabel(labels[i])
    axes[i].grid(True)
    axes[i].legend()

axes[-1].set_xlabel("Time [s]")

fig.suptitle(
    "Position Tracking"
)

plt.tight_layout()


# ============================================================
# Position tracking error
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time,
    error_history[:, 0],
    label="e_x"
)

plt.plot(
    time,
    error_history[:, 1],
    label="e_y"
)

plt.plot(
    time,
    error_history[:, 2],
    label="e_z"
)

plt.xlabel("Time [s]")
plt.ylabel("Position Error [m]")

plt.title(
    "Position Tracking Error"
)

plt.grid(True)
plt.legend()
plt.tight_layout()


# ============================================================
# Error magnitude
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    error_norm
)

plt.xlabel("Time [s]")
plt.ylabel("||e_x|| [m]")

plt.title(
    "Position Error Magnitude"
)

plt.grid(True)
plt.tight_layout()


# ============================================================
# Control inputs
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    thrust_history
)

plt.xlabel("Time [s]")
plt.ylabel("Thrust [N]")

plt.title(
    "Total Thrust Command"
)

plt.grid(True)
plt.tight_layout()


plt.figure(figsize=(10, 5))

plt.plot(
    time,
    moment_history[:, 0],
    label="Mx"
)

plt.plot(
    time,
    moment_history[:, 1],
    label="My"
)

plt.plot(
    time,
    moment_history[:, 2],
    label="Mz"
)

plt.xlabel("Time [s]")
plt.ylabel("Moment [N m]")

plt.title(
    "Body Moment Commands"
)

plt.grid(True)
plt.legend()
plt.tight_layout()


plt.show()