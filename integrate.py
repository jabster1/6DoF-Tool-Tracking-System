"""
integrate.py — Offline EULER-ANGLE orientation propagation from IMU CSVs.

Reads timestamped gyro samples from imu_read.cpp, subtracts the measured
biases (bx=0.228, by=0.075, bz=-0.045), and integrates roll/pitch/yaw
sequentially over the recording.

Test profile: 0 deg -> 90 deg -> 0 deg over ~15 s.
Results (baseline): 90.2 deg at the 90 deg hold; 21 deg pitch residual
after returning to 0 deg. This file is the baseline
"""

import csv
# Gyro biases from the stationary calibration run raw units
bias = {'gx': 0.228, 'gy': 0.075, 'gz': -0.045}
#running angle estimates x,y,z
roll = pitch = yaw = 0.0
prev_t, snap = None, None
with open('imu_log.csv') as f:
    for row in csv.DictReader(f):
        t = int(row['t_ms'])
        if prev_t is not None: #first row has no dt or prev timestamp
            #time between two consecutive rows
            dt = (t - prev_t) / 1000.0
            #(raw-bias); /16 -> deg/s; *dt -> deg turned;
            roll  += (float(row['gx']) - bias['gx']) / 16.0 * dt
            pitch += (float(row['gy']) - bias['gy']) / 16.0 * dt
            yaw   += (float(row['gz']) - bias['gz']) / 16.0 * dt
        if snap is None and t >= 6000:
            snap = (roll, pitch, yaw)
        prev_t = t
print(f"at 6s hold: roll={snap[0]:.1f} pitch={snap[1]:.1f} yaw={snap[2]:.1f} deg")
print(f"at end:     roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} deg")
