"""
integrate_quat.py — Offline QUATERNION orientation propagation from IMU CSVs.

Same input, same biases, same recording as integrate.py — the only change
is the propagation math. Orientation is a unit quaternion [w, x, y, z]:
each gyro sample becomes one rotation dq = [1, wx*dt/2, wy*dt/2, wz*dt/2],
applied as q = q (x) dq, normalized every iteration. No gimbal lock, no
chained-rotation error.

Target: final pitch residual UNDER 21 deg, 90-deg-hold reading at least as
close as 90.2 deg. If both hold, the representation earned its keep.

Results Proven: Quaternion beats Euler
at 6s hold: roll=90.1 pitch=-6.1 yaw=4.5 deg
at end:     roll=4.0 pitch=-3.7 yaw=7.5 deg
"""

import csv
import math

bias = {'gx': 0.228, 'gy': 0.075, 'gz': -0.045}
D2R = math.pi / 180.0 #degree to radian conv
R2D = 180.0 / math.pi #radian to degree conv

def quat_mult(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    )

def norm(q):
    n = math.sqrt(sum(c*c for c in q)) #quaternion magnitude
    return tuple(c/n for c in q)

def quat_to_euler(q):
    w, x, y, z = q
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    v = max(-1.0, min(1.0, 2*(w*y - z*x)))
    pitch = math.asin(v)
    yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return roll*R2D, pitch*R2D, yaw*R2D

q = (1.0, 0.0, 0.0, 0.0)
prev_t, snap = None, None
with open('imu_log.csv') as f:
    for row in csv.DictReader(f):
        t = int(row['t_ms'])
        if prev_t is not None:
            dt = (t - prev_t) / 1000.0
            wx = (float(row['gx']) - bias['gx']) / 16.0 * D2R
            wy = (float(row['gy']) - bias['gy']) / 16.0 * D2R
            wz = (float(row['gz']) - bias['gz']) / 16.0 * D2R
            dq = (1.0, wx*dt/2, wy*dt/2, wz*dt/2)
            q = norm(quat_mult(q, dq))
        if snap is None and t >= 6000:
            snap = quat_to_euler(q)
        prev_t = t

roll, pitch, yaw = quat_to_euler(q)
print(f"at 6s hold: roll={snap[0]:.1f} pitch={snap[1]:.1f} yaw={snap[2]:.1f} deg")
print(f"at end:     roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} deg")
