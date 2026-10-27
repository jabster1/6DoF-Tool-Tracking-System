"""
eskf_accel.py — Error-State Kalman Filter: gyro PREDICTION + accelerometer UPDATE.

Builds on eskf_predict.py. Runs two filters on the same log so you can compare:
  A) prediction only   (gyro integration — drifts, uncertainty grows forever)
  B) prediction + accel update (gravity corrects roll & pitch, bias gets estimated)

Usage:
    python3 eskf_accel.py imu_log.csv
Expects columns: t_ms, ax, ay, az, gx, gy, gz  (raw BNO055 units)
Log must START with the sensor still for >= CAL_SEC seconds.

Outputs:
    eskf_accel_out.csv   — per-step angles / 3-sigma / bias for both filters
    eskf_accel.png       — before/after drift plot + bias estimate plot

Key idea of the accel update:
    The accelerometer, when not being shaken, measures gravity. Gravity tells
    the filter "which way is down" in the sensor frame — an ABSOLUTE reference
    for roll and pitch that never drifts. We compare the gravity direction the
    filter EXPECTS (from its orientation estimate) to the one MEASURED, and
    correct. Yaw is a rotation AROUND gravity, so gravity can't see it.
"""

import csv
import math
import sys

import numpy as np

# ── Tunables ────────────────────────────────────────────────────────────────
LSB_PER_DPS = 16.0          # BNO055 gyro: 16 LSB = 1 deg/s
CAL_SEC = 2.0               # stationary window used for startup calibration
SIGMA_BG = 1e-5             # gyro bias random walk [rad/s/sqrt(s)]
SIGMA_B0 = 5e-4             # initial bias uncertainty [rad/s]
SIGMA_TH0 = math.radians(0.1)  # initial attitude uncertainty [rad]
ACCEL_GATE = 0.10           # skip accel update if |a| differs from 1 g by >10% (board moving)
ACCEL_NOISE_FLOOR = 0.01    # minimum accel direction noise [unit vector] — guards against overconfidence

DEG = 180.0 / math.pi
E3 = np.array([0.0, 0.0, 1.0])   # "up" in the world frame


# ── Math helpers ────────────────────────────────────────────────────────────
def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def quat_mul(p, q):
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_from_rotvec(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.5*v[0], 0.5*v[1], 0.5*v[2]])
    axis = v / angle
    s = math.sin(angle / 2)
    return np.array([math.cos(angle / 2), *(axis * s)])


def quat_to_rotmat(q):
    """Rotation matrix R (body -> world) from unit quaternion [w,x,y,z]."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll, pitch, yaw


def quat_from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


# ── Data ────────────────────────────────────────────────────────────────────
def load(path):
    t, g, a = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(int(row['t_ms']) / 1000.0)
            g.append([float(row['gx']), float(row['gy']), float(row['gz'])])
            a.append([float(row['ax']), float(row['ay']), float(row['az'])])
    gyro = np.array(g) / LSB_PER_DPS * math.pi / 180.0   # -> rad/s
    accel = np.array(a)    # raw; only its DIRECTION is used, so units don't matter
    return np.array(t), gyro, accel


# ── The filter ──────────────────────────────────────────────────────────────
class ESKF:
    def __init__(self, q0, bias0, sigma_g, sigma_a, g_ref, use_accel):
        self.q = q0.copy()
        self.b = bias0.copy()
        self.P = np.diag([SIGMA_TH0**2]*3 + [SIGMA_B0**2]*3)
        self.sigma_g = sigma_g
        self.R = np.eye(3) * sigma_a**2
        self.g_ref = g_ref
        self.use_accel = use_accel
        self.n_updates = 0

    def predict(self, gyro, dt):
        w = gyro - self.b
        # nominal state
        self.q = quat_mul(self.q, quat_from_rotvec(w * dt))
        self.q /= np.linalg.norm(self.q)
        # error-state covariance
        F = np.eye(6)
        F[0:3, 0:3] = np.eye(3) - skew(w) * dt
        F[0:3, 3:6] = -np.eye(3) * dt
        Qd = np.diag([self.sigma_g**2 * dt]*3 + [SIGMA_BG**2 * dt]*3)
        self.P = F @ self.P @ F.T + Qd

    def update_accel(self, accel):
        if not self.use_accel:
            return
        norm = np.linalg.norm(accel)
        # Gate: only trust the accelerometer when it's measuring ~1 g (i.e. not being shaken)
        if abs(norm / self.g_ref - 1.0) > ACCEL_GATE:
            return

        z = accel / norm                       # measured "up" direction in sensor frame
        Rbw = quat_to_rotmat(self.q)
        h = Rbw.T @ E3                         # predicted "up" direction in sensor frame

        # Jacobian of h w.r.t. the error state [dtheta, dbias]:
        #   h(q ⊗ exp(dtheta)) ≈ h + [h×] dtheta ;  bias doesn't affect gravity directly
        H = np.zeros((3, 6))
        H[:, 0:3] = skew(h)

        y = z - h                              # innovation: measured minus predicted
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)    # Kalman gain (6x3)
        dx = K @ y                             # error-state correction

        # Covariance update (Joseph form — numerically stable, keeps P symmetric)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Inject the error into the nominal state, then the error resets to zero
        self.q = quat_mul(self.q, quat_from_rotvec(dx[0:3]))
        self.q /= np.linalg.norm(self.q)
        self.b = self.b + dx[3:6]
        self.n_updates += 1


# ── Main ────────────────────────────────────────────────────────────────────
def main(path):
    t, gyro, accel = load(path)
    t = t - t[0]

    still = t < CAL_SEC
    if still.sum() < 20:
        sys.exit(f"Need >= {CAL_SEC}s of stationary data at the start of the log.")

    # Startup calibration (same as eskf_predict.py) + accel reference
    bias0 = gyro[still].mean(axis=0)
    sigma_g = gyro[still].std(axis=0).mean()
    a_still = accel[still]
    g_ref = np.linalg.norm(a_still, axis=1).mean()          # what "1 g" reads as, in raw units
    a_dir = a_still / np.linalg.norm(a_still, axis=1, keepdims=True)
    sigma_a = max(a_dir.std(axis=0).mean(), ACCEL_NOISE_FLOOR)

    # Initial roll/pitch from gravity (yaw = 0: gravity can't tell heading)
    am = a_still.mean(axis=0)
    roll0 = math.atan2(am[1], am[2])
    pitch0 = math.atan2(-am[0], math.hypot(am[1], am[2]))
    q0 = quat_from_euler(roll0, pitch0, 0.0)

    print(f"Samples: {len(t)}  duration: {t[-1]:.1f}s  rate: {1/np.median(np.diff(t)):.1f} Hz")
    print(f"Gyro bias (rad/s): {bias0}   sigma_g: {sigma_g:.2e} rad/s")
    print(f"Accel '1 g' reads as: {g_ref:.1f} raw   sigma_a (direction): {sigma_a:.3f}")
    print(f"Initial tilt from gravity: roll={roll0*DEG:.2f} pitch={pitch0*DEG:.2f} deg")

    filt_pred = ESKF(q0, bias0, sigma_g, sigma_a, g_ref, use_accel=False)
    filt_acc = ESKF(q0, bias0, sigma_g, sigma_a, g_ref, use_accel=True)
    e0 = np.array([roll0, pitch0, 0.0])

    rows = []
    for k in range(1, len(t)):
        dt = t[k] - t[k-1]
        if dt <= 0:
            continue
        for f in (filt_pred, filt_acc):
            f.predict(gyro[k-1], dt)
            f.update_accel(accel[k])

        row = [t[k]]
        for f in (filt_pred, filt_acc):
            ang = (np.array(quat_to_euler(f.q)) - e0) * DEG     # change since start
            s3 = 3 * np.sqrt(np.diag(f.P)[0:3]) * DEG
            row += [*ang, *s3]
        row += list(filt_acc.b * DEG)                            # bias estimate, deg/s
        rows.append(row)

    out = np.array(rows)
    hdr = ['t',
           'pred_roll', 'pred_pitch', 'pred_yaw', 'pred_roll_3s', 'pred_pitch_3s', 'pred_yaw_3s',
           'acc_roll', 'acc_pitch', 'acc_yaw', 'acc_roll_3s', 'acc_pitch_3s', 'acc_yaw_3s',
           'bias_x_dps', 'bias_y_dps', 'bias_z_dps']
    with open('eskf_accel_out.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(hdr)
        wr.writerows(out.round(6).tolist())

    end = out[-1]
    print(f"\nAccel updates applied: {filt_acc.n_updates} / {len(t)-1}")
    print(f"\nAt end ({end[0]:.1f}s)          roll     pitch    yaw   (deg drift from start)")
    print(f"  Prediction only:   {end[1]:7.2f}  {end[2]:7.2f}  {end[3]:7.2f}"
          f"   ±3σ: {end[4]:.2f} / {end[5]:.2f} / {end[6]:.2f}")
    print(f"  + Accel update:    {end[7]:7.2f}  {end[8]:7.2f}  {end[9]:7.2f}"
          f"   ±3σ: {end[10]:.2f} / {end[11]:.2f} / {end[12]:.2f}")
    print(f"\nBias: startup {bias0*DEG} deg/s  ->  final estimate {filt_acc.b*DEG} deg/s")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        names = ['roll', 'pitch', 'yaw']
        for i, name in enumerate(names):
            ax = axes[i]
            ax.fill_between(out[:, 0], -out[:, 4+i], out[:, 4+i],
                            color='tab:red', alpha=0.12, label='±3σ prediction only')
            ax.fill_between(out[:, 0], -out[:, 10+i], out[:, 10+i],
                            color='tab:blue', alpha=0.25, label='±3σ with accel update')
            ax.plot(out[:, 0], out[:, 1+i], color='tab:red', label='prediction only')
            ax.plot(out[:, 0], out[:, 7+i], color='tab:blue', label='with accel update')
            ax.set_ylabel(f'{name} (deg)')
            ax.legend(loc='upper left', fontsize=7, ncol=2)
        ax = axes[3]
        for j, (lbl, c) in enumerate(zip(['x', 'y', 'z'], ['tab:green', 'tab:orange', 'tab:purple'])):
            ax.plot(out[:, 0], out[:, 13+j], color=c, label=f'gyro bias {lbl} estimate')
        ax.set_ylabel('bias (deg/s)')
        ax.set_xlabel('time (s)')
        ax.legend(loc='upper left', fontsize=7, ncol=3)
        fig.suptitle('ES-EKF: gyro prediction vs. prediction + accelerometer update')
        fig.tight_layout()
        fig.savefig('eskf_accel.png', dpi=130)
        print("Saved eskf_accel.png")
    except ImportError:
        print("matplotlib not installed — plot eskf_accel_out.csv however you like.")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'imu_log.csv')
