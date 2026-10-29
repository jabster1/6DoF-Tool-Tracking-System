/*
 * eskf.h — Error-State Kalman Filter for IMU orientation (C++ port of eskf_accel.py)
 *
 *   Nominal state : q (orientation quaternion, body->world), b (gyro bias, rad/s)
 *   Error state   : [dtheta(3), dbias(3)]  — only its covariance P (6x6) is stored
 *
 *   predict(gyro, dt)   gyro integration; uncertainty grows   (P = F P F^T + Qd)
 *   update_accel(acc)   gravity corrects roll/pitch; bias becomes observable
 *
 * Header-only. Include it in the offline runner, the live program, and later
 * in the tool tracker.
 */
#pragma once
#include <cmath>
#include <vector>
#include "mat.h"

namespace eskf {

// ── Tunables (same values as eskf_accel.py) ────────────────────────────────
struct Config {
    double lsb_per_dps       = 16.0;     // BNO055 gyro: 16 LSB = 1 deg/s
    double cal_sec           = 2.0;      // stationary window for startup calibration
    double sigma_bg          = 1e-5;     // gyro bias random walk [rad/s/sqrt(s)]
    double sigma_b0          = 5e-4;     // initial bias uncertainty [rad/s]
    double sigma_th0         = 0.1 * M_PI / 180.0;  // initial attitude uncertainty [rad]
    double accel_gate        = 0.10;     // skip accel update if |a| is >10% off 1 g
    double accel_noise_floor = 0.01;     // min accel direction noise (unit vector)
};

inline Vec3 raw_gyro_to_rads(const Vec3& raw, const Config& cfg) {
    return raw * (M_PI / 180.0 / cfg.lsb_per_dps);
}

// ── Startup calibration result ──────────────────────────────────────────────
struct Calibration {
    Vec3   gyro_bias;   // rad/s
    double sigma_g = 0; // gyro white noise, rad/s       -> Q
    double g_ref   = 0; // what "1 g" reads as, raw units -> accel gate
    double sigma_a = 0; // accel direction noise          -> R
    double roll0 = 0, pitch0 = 0;   // initial tilt from gravity, rad
    Quat   q0;          // initial orientation (yaw = 0)
};

// Compute calibration from samples taken while the sensor is STILL.
// gyro in rad/s, accel in raw units (only direction is used).
inline Calibration calibrate(const std::vector<Vec3>& gyro,
                             const std::vector<Vec3>& accel,
                             const Config& cfg) {
    Calibration c;
    const int n = (int)gyro.size();

    // gyro bias = mean, noise = population std averaged over axes (same as numpy .std())
    Vec3 mean_g, mean_a, mean_dir;
    for (int i = 0; i < n; ++i) mean_g = mean_g + gyro[i];
    mean_g = mean_g * (1.0 / n);
    Vec3 var_g;
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) {
        double d = gyro[i][k] - mean_g[k]; var_g[k] += d * d;
    }
    c.gyro_bias = mean_g;
    c.sigma_g = (std::sqrt(var_g[0]/n) + std::sqrt(var_g[1]/n) + std::sqrt(var_g[2]/n)) / 3.0;

    // accel: 1 g reference, direction noise, initial tilt
    std::vector<Vec3> dirs(n);
    double norm_sum = 0;
    for (int i = 0; i < n; ++i) {
        double nn = accel[i].norm();
        norm_sum += nn;
        dirs[i] = accel[i] * (1.0 / nn);
        mean_a = mean_a + accel[i];
        mean_dir = mean_dir + dirs[i];
    }
    c.g_ref = norm_sum / n;
    mean_a = mean_a * (1.0 / n);
    mean_dir = mean_dir * (1.0 / n);
    Vec3 var_d;
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) {
        double d = dirs[i][k] - mean_dir[k]; var_d[k] += d * d;
    }
    double sa = (std::sqrt(var_d[0]/n) + std::sqrt(var_d[1]/n) + std::sqrt(var_d[2]/n)) / 3.0;
    c.sigma_a = std::max(sa, cfg.accel_noise_floor);

    c.roll0  = std::atan2(mean_a[1], mean_a[2]);
    c.pitch0 = std::atan2(-mean_a[0], std::hypot(mean_a[1], mean_a[2]));
    c.q0 = quat_from_euler(c.roll0, c.pitch0, 0.0);
    return c;
}

// ── The filter ──────────────────────────────────────────────────────────────
class ESKF {
public:
    Quat q;            // nominal orientation
    Vec3 b;            // nominal gyro bias (rad/s)
    Mat6 P;            // error-state covariance
    int  n_updates = 0;

    ESKF() = default;
    ESKF(const Calibration& cal, const Config& cfg, bool use_accel = true) { init(cal, cfg, use_accel); }

    void init(const Calibration& cal, const Config& cfg, bool use_accel = true) {
        cfg_ = cfg;
        use_accel_ = use_accel;
        q = cal.q0;
        b = cal.gyro_bias;
        sigma_g_ = cal.sigma_g;
        g_ref_ = cal.g_ref;
        R_ = Mat3::Identity() * (cal.sigma_a * cal.sigma_a);
        P = Mat6::Zero();
        for (int i = 0; i < 3; ++i) P(i, i)     = cfg.sigma_th0 * cfg.sigma_th0;
        for (int i = 3; i < 6; ++i) P(i, i)     = cfg.sigma_b0  * cfg.sigma_b0;
        n_updates = 0;
    }

    // gyro in rad/s, dt in seconds
    void predict(const Vec3& gyro, double dt) {
        Vec3 w = gyro - b;                               // bias-compensated rate

        // nominal state: q <- q ⊗ exp(w dt)
        q = q * quat_from_rotvec(w * dt);
        q.normalize();

        // error-state transition
        Mat6 F = Mat6::Identity();
        F.set_block(0, 0, Mat3::Identity() - skew(w) * dt);
        F.set_block(0, 3, Mat3::Identity() * (-dt));

        Mat6 Qd;
        for (int i = 0; i < 3; ++i) Qd(i, i) = sigma_g_ * sigma_g_ * dt;
        for (int i = 3; i < 6; ++i) Qd(i, i) = cfg_.sigma_bg * cfg_.sigma_bg * dt;

        P = F * P * F.T() + Qd;
    }

    // accel in any units (direction only). Returns true if the update was applied.
    bool update_accel(const Vec3& accel) {
        if (!use_accel_) return false;
        double norm = accel.norm();
        if (std::abs(norm / g_ref_ - 1.0) > cfg_.accel_gate) return false;   // board moving

        Vec3 z = accel * (1.0 / norm);                  // measured "up" (sensor frame)
        Vec3 h = quat_to_rotmat(q).T() * vec3(0, 0, 1); // predicted "up" (sensor frame)

        Mat<3, 6> H;                                    // d h / d[dtheta, dbias]
        H.set_block(0, 0, skew(h));                     // bias columns stay 0

        Vec3 y = z - h;                                 // innovation
        Mat3 S = H * P * H.T() + R_;
        Mat<6, 3> K = P * H.T() * inverse(S);           // Kalman gain
        Mat<6, 1> dx = K * y;                           // error-state correction

        // Joseph-form covariance update (keeps P symmetric / positive)
        Mat6 I_KH = Mat6::Identity() - K * H;
        P = I_KH * P * I_KH.T() + K * R_ * K.T();

        // inject correction into nominal state; error resets to zero
        q = q * quat_from_rotvec(dx.block<3, 1>(0, 0));
        q.normalize();
        b = b + dx.block<3, 1>(3, 0);
        ++n_updates;
        return true;
    }

    Vec3 euler() const { return quat_to_euler(q); }              // rad
    Vec3 sigma3_deg() const {                                     // ±3σ attitude, deg
        return vec3(3 * std::sqrt(P(0,0)), 3 * std::sqrt(P(1,1)), 3 * std::sqrt(P(2,2))) * (180.0 / M_PI);
    }

private:
    Config cfg_;
    bool   use_accel_ = true;
    double sigma_g_ = 0, g_ref_ = 1;
    Mat3   R_;
};

}  // namespace eskf
