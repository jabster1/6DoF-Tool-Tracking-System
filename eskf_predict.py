"""
eskf_predict.py — Error-State Kalman Filter, PREDICTION ONLY, on real BNO055 gyro logs.

Upgrade of integrate.py:
  * bias is estimated from the first CAL_SEC of each log (bias changes every power-up)
  * orientation is integrated as a QUATERNION (correct for combined rotations,
    unlike adding body rates to roll/pitch/yaw directly)
  * a 6-state error covariance P = [dtheta(3), dbias(3)] is propagated with Q,
    so you get a 3-sigma uncertainty bound to plot against the actual drift

Usage:
    python3 eskf_predict.py imu_log.csv
Expects columns: t_ms, gx, gy, gz  (raw BNO055 LSB, 16 LSB = 1 deg/s)
Log should START with the sensor sitting still for >= CAL_SEC seconds.

Outputs:
    eskf_out.csv   — t, roll/pitch/yaw (deg), 3-sigma (deg) per axis
    eskf_drift.png — angle drift with 3-sigma bounds (if matplotlib is installed)
"""

import csv
import math
import sys

import numpy as np

# ── Tunables ────────────────────────────────────────────────────────────────
LSB_PER_DPS = 16.0          # BNO055 default gyro scale
CAL_SEC = 2.0               # stationary window at start used for bias + noise
SIGMA_BG = 1e-5             # gyro bias random walk [rad/s/sqrt(s)] — small: bias ~constant in-run
SIGMA_B0 = 5e-4             # initial bias uncertainty [rad/s] (~ measured residual 3.1e-4)
SIGMA_TH0 = math.radians(0.1)  # initial attitude uncertainty [rad]

DEG = 180.0 / math.pi


def raw_to_rads(raw):
    return raw / LSB_PER_DPS * math.pi / 180.0


def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def quat_mul(p, q):
    """Hamilton product, [w, x, y, z]."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_from_rotvec(v):
    """Exact exponential map: rotation vector (rad) -> unit quaternion."""
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.5*v[0], 0.5*v[1], 0.5*v[2]])
    axis = v / angle
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), *(axis * s)])


def quat_to_euler(q):
    """ZYX (yaw-pitch-roll) Euler angles in radians."""
    w, x, y, z = q
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def load(path):
    t, g = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(int(row['t_ms']) / 1000.0)
            g.append([float(row['gx']), float(row['gy']), float(row['gz'])])
    return np.array(t), raw_to_rads(np.array(g))


def main(path):
    t, gyro = load(path)
    t = t - t[0]

    # ── 1. Startup calibration: bias + white-noise sigma from still window ──
    still = t < CAL_SEC
    if still.sum() < 20:
        sys.exit(f"Need >= {CAL_SEC}s of stationary data at the start of the log.")
    bias = gyro[still].mean(axis=0)
    sigma_g = gyro[still].std(axis=0).mean()     # rad/s, gyro white noise
    rate_hz = 1.0 / np.median(np.diff(t))

    print(f"Samples: {len(t)}  duration: {t[-1]:.1f}s  rate: {rate_hz:.1f} Hz")
    print(f"Bias (rad/s): {bias}   (raw LSB: {bias / math.pi * 180 * LSB_PER_DPS})")
    print(f"sigma_g (gyro noise): {sigma_g:.2e} rad/s  -> Q justification")

    # ── 2. Filter state ──────────────────────────────────────────────────────
    q = np.array([1.0, 0.0, 0.0, 0.0])           # nominal orientation
    b = bias.copy()                               # nominal bias
    P = np.diag([SIGMA_TH0**2]*3 + [SIGMA_B0**2]*3)

    rows = []
    for k in range(1, len(t)):
        dt = t[k] - t[k-1]
        if dt <= 0:
            continue
        w = gyro[k-1] - b                         # bias-compensated rate

        # nominal: q <- q ⊗ exp(w dt), normalized
        q = quat_mul(q, quat_from_rotvec(w * dt))
        q /= np.linalg.norm(q)

        # error state: dtheta_dot = -[w x] dtheta - dbias ;  dbias_dot = 0 (+ noise)
        F = np.eye(6)
        F[0:3, 0:3] = np.eye(3) - skew(w) * dt
        F[0:3, 3:6] = -np.eye(3) * dt
        Qd = np.diag([sigma_g**2 * dt]*3 + [SIGMA_BG**2 * dt]*3)
        P = F @ P @ F.T + Qd

        r, p, y = quat_to_euler(q)
        s3 = 3 * np.sqrt(np.diag(P)[0:3]) * DEG
        rows.append([t[k], r*DEG, p*DEG, y*DEG, *s3])

    out = np.array(rows)
    with open('eskf_out.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['t', 'roll', 'pitch', 'yaw', 'roll_3s', 'pitch_3s', 'yaw_3s'])
        wr.writerows(out.round(5).tolist())

    end = out[-1]
    print(f"\nAt end ({end[0]:.1f}s): roll={end[1]:.2f} pitch={end[2]:.2f} yaw={end[3]:.2f} deg")
    print(f"3-sigma bound:          ±{end[4]:.2f}     ±{end[5]:.2f}      ±{end[6]:.2f} deg")
    print("If the log was stationary, |angle| should sit inside the 3-sigma bound.")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for i, name in enumerate(['roll', 'pitch', 'yaw']):
            ax = axes[i]
            ax.plot(out[:, 0], out[:, 1+i], label=f'{name} (gyro integration)')
            ax.fill_between(out[:, 0], -out[:, 4+i], out[:, 4+i],
                            alpha=0.2, label='±3σ (filter covariance)')
            ax.set_ylabel('deg')
            ax.legend(loc='upper left', fontsize=8)
        axes[-1].set_xlabel('time (s)')
        fig.suptitle('ES-EKF prediction: gyro drift vs. predicted uncertainty')
        fig.tight_layout()
        fig.savefig('eskf_drift.png', dpi=130)
        print("Saved eskf_drift.png")
    except ImportError:
        print("matplotlib not installed — plot eskf_out.csv however you like.")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'imu_log.csv')