/*
 * mat.h — tiny fixed-size matrix + quaternion helpers for the ESKF.
 *
 * Zero dependencies (no Eigen needed). Written so the filter code reads like
 * the equations:   P = F * P * F.T() + Qd;
 *
 * Mat<R,C> is row-major. Vec3 is Mat<3,1>.
 * If you later switch to Eigen, every operation here has a 1:1 equivalent:
 *   Mat<6,6>          -> Eigen::Matrix<double,6,6>
 *   A.T()             -> A.transpose()
 *   A.set_block(r,c,B)-> A.block<..>(r,c) = B
 *   Quat              -> Eigen::Quaterniond  (careful: Eigen stores x,y,z,w)
 */
#pragma once
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
