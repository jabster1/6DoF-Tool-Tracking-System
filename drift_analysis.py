import csv, math

def quat_mult(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw]

def quat_angle(a, b):
    # angle between two orientations, in degrees
    d = min(1.0, abs(sum(x*y for x, y in zip(a, b))))
    return 2.0 * math.degrees(math.acos(d))

rows = []
with open('imu_log.csv') as f:
    for r in csv.DictReader(f):
        rows.append((int(r['t_ms']), float(r['gx']), float(r['gy']), float(r['gz'])))
t_end = rows[-1][0]

# First 5 s bookend -> fresh bias (better than the Sep 29 numbers: 12 days stale)
b = [r for r in rows if r[0] < 5000]
bias = [sum(r[i] for r in b)/len(b) for i in (1, 2, 3)]
# Last 5 s bookend -> stillness proof (should match bias: rig undisturbed)
e = [r for r in rows if r[0] > t_end - 5000]
tail = [sum(r[i] for r in e)/len(e) for i in (1, 2, 3)]
print(f"fresh bias (raw): gx={bias[0]:.3f} gy={bias[1]:.3f} gz={bias[2]:.3f}")
print(f"last-5s mean  (raw): gx={tail[0]:.3f} gy={tail[1]:.3f} gz={tail[2]:.3f}")

# Propagate with the fresh bias; drift = angle from the 5 s mark
q, q_ref, snaps, prev_t = [1.0, 0.0, 0.0, 0.0], None, {}, None
for t, gx, gy, gz in rows:
    if prev_t is not None:
        dt = (t - prev_t) / 1000.0
        wx = (gx-bias[0])/16.0*math.pi/180.0
        wy = (gy-bias[1])/16.0*math.pi/180.0
        wz = (gz-bias[2])/16.0*math.pi/180.0
        om = math.sqrt(wx*wx + wy*wy + wz*wz)
        th = om*dt
        if th > 1e-12:
            s = math.sin(th/2)/om
            q = quat_mult(q, [math.cos(th/2), s*wx, s*wy, s*wz])
            n = math.sqrt(sum(v*v for v in q)); q = [v/n for v in q]
    if q_ref is None and t >= 5000:
        q_ref = list(q)
    for mark in (15000, 35000, 65000):
        if mark not in snaps and t >= mark:
            snaps[mark] = quat_angle(q, q_ref)
    prev_t = t

for mark in (15000, 35000, 65000):
    print(f"drift at +{(mark-5000)//1000}s: {snaps[mark]:.2f} deg")
