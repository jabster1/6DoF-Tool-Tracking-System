/*
 * eskf_replay.cpp — Error-State Kalman Filter (C++ port of eskf_accel.py), single file.
 *
 *   Build:  g++ -O2 -std=c++17 eskf_replay.cpp -o eskf_replay
 *   Run:    ./eskf_replay imu_log.csv
 *
 * Input : CSV with columns t_ms, ax, ay, az, gx, gy, gz (raw BNO055 units).
 *         First 2 s must be still (startup calibration).
 * Output: same summary as eskf_accel.py, plus eskf_cpp_out.csv
 *         (same columns as eskf_accel_out.csv, so the two diff directly).
 *
 * Runs two filters on the same data: prediction only, and prediction + accel update.
 *
 * Sections (same order as the Python):
 *   1. Matrix + quaternion helpers
 *   2. Config (tunables) + startup calibration
 *   3. ESKF class: predict() and update_accel()
 *   4. CSV loading + main loop
 */

// ═══ 1. Matrix + quaternion helpers ═══════════════════════════════════════
#include <cmath>
#include <algorithm>

namespace eskf {

template <int R, int C>
struct Mat {
    double a[R * C] = {0};

    double&       operator()(int r, int c)       { return a[r * C + c]; }
    const double& operator()(int r, int c) const { return a[r * C + c]; }
    // vector-style access for column vectors
    double&       operator[](int i)       { return a[i]; }
    const double& operator[](int i) const { return a[i]; }

    static Mat Zero() { return Mat{}; }
    static Mat Identity() {
        static_assert(R == C, "Identity needs a square matrix");
        Mat m; for (int i = 0; i < R; ++i) m(i, i) = 1.0; return m;
    }

    Mat<C, R> T() const {
        Mat<C, R> t;
        for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) t(c, r) = (*this)(r, c);
        return t;
    }

    // copy a smaller matrix B into this one at (r0, c0) — Python's F[0:3, 3:6] = B
    template <int R2, int C2>
    void set_block(int r0, int c0, const Mat<R2, C2>& B) {
        for (int r = 0; r < R2; ++r) for (int c = 0; c < C2; ++c) (*this)(r0 + r, c0 + c) = B(r, c);
    }
    // read a sub-matrix — Python's dx[0:3]
    template <int R2, int C2>
    Mat<R2, C2> block(int r0, int c0) const {
        Mat<R2, C2> B;
        for (int r = 0; r < R2; ++r) for (int c = 0; c < C2; ++c) B(r, c) = (*this)(r0 + r, c0 + c);
        return B;
    }

    double norm() const { double s = 0; for (double v : a) s += v * v; return std::sqrt(s); }
};

template <int R, int C>
Mat<R, C> operator+(const Mat<R, C>& A, const Mat<R, C>& B) { Mat<R, C> M; for (int i = 0; i < R*C; ++i) M.a[i] = A.a[i] + B.a[i]; return M; }
template <int R, int C>
Mat<R, C> operator-(const Mat<R, C>& A, const Mat<R, C>& B) { Mat<R, C> M; for (int i = 0; i < R*C; ++i) M.a[i] = A.a[i] - B.a[i]; return M; }
template <int R, int C>
Mat<R, C> operator*(const Mat<R, C>& A, double s) { Mat<R, C> M; for (int i = 0; i < R*C; ++i) M.a[i] = A.a[i] * s; return M; }
template <int R, int C>
Mat<R, C> operator*(double s, const Mat<R, C>& A) { return A * s; }

template <int R, int K, int C>
Mat<R, C> operator*(const Mat<R, K>& A, const Mat<K, C>& B) {
    Mat<R, C> M;
    for (int r = 0; r < R; ++r)
        for (int k = 0; k < K; ++k) {
            double v = A(r, k);
            for (int c = 0; c < C; ++c) M(r, c) += v * B(k, c);
        }
    return M;
}

using Vec3 = Mat<3, 1>;
using Mat3 = Mat<3, 3>;
using Mat6 = Mat<6, 6>;

inline Vec3 vec3(double x, double y, double z) { Vec3 v; v[0] = x; v[1] = y; v[2] = z; return v; }

// [w×] — skew-symmetric matrix so that skew(a) * b == a × b
inline Mat3 skew(const Vec3& w) {
    Mat3 S;
    S(0,1) = -w[2]; S(0,2) =  w[1];
    S(1,0) =  w[2]; S(1,2) = -w[0];
    S(2,0) = -w[1]; S(2,1) =  w[0];
    return S;
}

// 3x3 inverse (closed form via adjugate). S in the Kalman update is 3x3.
inline Mat3 inverse(const Mat3& m) {
    double a=m(0,0), b=m(0,1), c=m(0,2), d=m(1,0), e=m(1,1), f=m(1,2), g=m(2,0), h=m(2,1), i=m(2,2);
    double A =  (e*i - f*h), B = -(d*i - f*g), C =  (d*h - e*g);
    double det = a*A + b*B + c*C;
    Mat3 inv;
    inv(0,0) = A;            inv(0,1) = -(b*i - c*h); inv(0,2) =  (b*f - c*e);
    inv(1,0) = B;            inv(1,1) =  (a*i - c*g); inv(1,2) = -(a*f - c*d);
    inv(2,0) = C;            inv(2,1) = -(a*h - b*g); inv(2,2) =  (a*e - b*d);
    return inv * (1.0 / det);
}

// ── Quaternion  [w, x, y, z], Hamilton convention ───────────────────────────
struct Quat {
    double w = 1, x = 0, y = 0, z = 0;

    void normalize() { double n = std::sqrt(w*w + x*x + y*y + z*z); w/=n; x/=n; y/=n; z/=n; }
};

// Hamilton product p ⊗ q  (apply rotation q after p, in p's body frame)
inline Quat operator*(const Quat& p, const Quat& q) {
    return {
        p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z,
        p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y,
        p.w*q.y - p.x*q.z + p.y*q.w + p.z*q.x,
        p.w*q.z + p.x*q.y - p.y*q.x + p.z*q.w,
    };
}

// rotation vector (axis * angle, rad) -> quaternion  (exponential map)
inline Quat quat_from_rotvec(const Vec3& v) {
    double angle = v.norm();
    if (angle < 1e-12) return {1.0, 0.5*v[0], 0.5*v[1], 0.5*v[2]};
    double s = std::sin(angle / 2) / angle;
    return {std::cos(angle / 2), v[0]*s, v[1]*s, v[2]*s};
}

// rotation matrix R (body -> world)
inline Mat3 quat_to_rotmat(const Quat& q) {
    double w=q.w, x=q.x, y=q.y, z=q.z;
    Mat3 R;
    R(0,0) = 1 - 2*(y*y + z*z); R(0,1) = 2*(x*y - w*z);     R(0,2) = 2*(x*z + w*y);
    R(1,0) = 2*(x*y + w*z);     R(1,1) = 1 - 2*(x*x + z*z); R(1,2) = 2*(y*z - w*x);
    R(2,0) = 2*(x*z - w*y);     R(2,1) = 2*(y*z + w*x);     R(2,2) = 1 - 2*(x*x + y*y);
    return R;
}

// ZYX Euler: returns (roll, pitch, yaw) in radians
inline Vec3 quat_to_euler(const Quat& q) {
    double w=q.w, x=q.x, y=q.y, z=q.z;
    double roll  = std::atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y));
    double pitch = std::asin(std::clamp(2*(w*y - z*x), -1.0, 1.0));
    double yaw   = std::atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z));
    return vec3(roll, pitch, yaw);
}

inline Quat quat_from_euler(double roll, double pitch, double yaw) {
    double cr = std::cos(roll/2),  sr = std::sin(roll/2);
    double cp = std::cos(pitch/2), sp = std::sin(pitch/2);
    double cy = std::cos(yaw/2),   sy = std::sin(yaw/2);
    return {
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    };
}

}  // namespace eskf

// ═══ 2-3. Config, calibration, ESKF ═══════════════════════════════════════
#include <cmath>
#include <vector>

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

// ═══ 4. CSV loading + main loop ═══════════════════════════════════════════
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>


using namespace eskf;

struct Sample { double t; Vec3 gyro; Vec3 accel; };

static std::vector<Sample> load_csv(const std::string& path, const Config& cfg) {
    std::ifstream f(path);
    if (!f) { std::cerr << "Cannot open " << path << "\n"; std::exit(1); }

    std::string line;
    std::getline(f, line);                       // header -> column index map
    std::map<std::string, int> col;
    { std::stringstream ss(line); std::string name; int i = 0;
      while (std::getline(ss, name, ',')) { while (!name.empty() && (name.back()=='\r'||name.back()==' ')) name.pop_back(); col[name] = i++; } }
    for (const char* need : {"t_ms","ax","ay","az","gx","gy","gz"})
        if (!col.count(need)) { std::cerr << "Missing column: " << need << "\n"; std::exit(1); }

    std::vector<Sample> out;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> v; std::stringstream ss(line); std::string cell;
        while (std::getline(ss, cell, ',')) v.push_back(std::stod(cell));
        Sample s;
        s.t = v[col["t_ms"]] / 1000.0;
        s.gyro  = raw_gyro_to_rads(vec3(v[col["gx"]], v[col["gy"]], v[col["gz"]]), cfg);
        s.accel = vec3(v[col["ax"]], v[col["ay"]], v[col["az"]]);
        out.push_back(s);
    }
    return out;
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "imu_log.csv";
    Config cfg;
    auto data = load_csv(path, cfg);
    if (data.size() < 50) { std::cerr << "Not enough samples\n"; return 1; }
    const double t0 = data[0].t;
    for (auto& s : data) s.t -= t0;

    // ── Startup calibration on the first cal_sec of (still) data ──────────
    std::vector<Vec3> g_still, a_still;
    for (const auto& s : data) if (s.t < cfg.cal_sec) { g_still.push_back(s.gyro); a_still.push_back(s.accel); }
    if (g_still.size() < 20) { std::cerr << "Need >= " << cfg.cal_sec << "s still at start\n"; return 1; }
    Calibration cal = calibrate(g_still, a_still, cfg);

    const double DEG = 180.0 / M_PI;
    std::printf("Samples: %zu  duration: %.1fs\n", data.size(), data.back().t);
    std::printf("Gyro bias (rad/s): [%.8f %.8f %.8f]   sigma_g: %.2e rad/s\n",
                cal.gyro_bias[0], cal.gyro_bias[1], cal.gyro_bias[2], cal.sigma_g);
    std::printf("Accel '1 g' reads as: %.1f raw   sigma_a (direction): %.3f\n", cal.g_ref, cal.sigma_a);
    std::printf("Initial tilt from gravity: roll=%.2f pitch=%.2f deg\n", cal.roll0*DEG, cal.pitch0*DEG);

    ESKF pred(cal, cfg, /*use_accel=*/false);
    ESKF acc (cal, cfg, /*use_accel=*/true);
    const Vec3 e0 = vec3(cal.roll0, cal.pitch0, 0.0);

    std::ofstream out("eskf_cpp_out.csv");
    out << "t,pred_roll,pred_pitch,pred_yaw,pred_roll_3s,pred_pitch_3s,pred_yaw_3s,"
           "acc_roll,acc_pitch,acc_yaw,acc_roll_3s,acc_pitch_3s,acc_yaw_3s,"
           "bias_x_dps,bias_y_dps,bias_z_dps\n";

    // ── Main loop: predict with gyro[k-1] over dt, then update with accel[k] ──
    for (size_t k = 1; k < data.size(); ++k) {
        double dt = data[k].t - data[k-1].t;
        if (dt <= 0) continue;
        for (ESKF* f : {&pred, &acc}) {
            f->predict(data[k-1].gyro, dt);
            f->update_accel(data[k].accel);
        }
        out << data[k].t;
        for (ESKF* f : {&pred, &acc}) {
            Vec3 ang = (f->euler() - e0) * DEG, s3 = f->sigma3_deg();
            out << ',' << ang[0] << ',' << ang[1] << ',' << ang[2]
                << ',' << s3[0]  << ',' << s3[1]  << ',' << s3[2];
        }
        out << ',' << acc.b[0]*DEG << ',' << acc.b[1]*DEG << ',' << acc.b[2]*DEG << '\n';
    }

    Vec3 ap = (pred.euler() - e0) * DEG, sp = pred.sigma3_deg();
    Vec3 aa = (acc.euler()  - e0) * DEG, sa = acc.sigma3_deg();
    std::printf("\nAccel updates applied: %d / %zu\n", acc.n_updates, data.size() - 1);
    std::printf("\nAt end (%.1fs)          roll     pitch    yaw   (deg drift from start)\n", data.back().t);
    std::printf("  Prediction only:   %7.2f  %7.2f  %7.2f   ±3σ: %.2f / %.2f / %.2f\n", ap[0], ap[1], ap[2], sp[0], sp[1], sp[2]);
    std::printf("  + Accel update:    %7.2f  %7.2f  %7.2f   ±3σ: %.2f / %.2f / %.2f\n", aa[0], aa[1], aa[2], sa[0], sa[1], sa[2]);
    std::printf("\nBias: startup [%.5f %.5f %.5f] deg/s  ->  final [%.5f %.5f %.5f] deg/s\n",
                cal.gyro_bias[0]*DEG, cal.gyro_bias[1]*DEG, cal.gyro_bias[2]*DEG,
                acc.b[0]*DEG, acc.b[1]*DEG, acc.b[2]*DEG);
    std::printf("Wrote eskf_cpp_out.csv\n");
    return 0;
}
