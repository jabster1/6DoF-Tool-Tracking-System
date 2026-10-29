/*
 * camera_motion.h — turn IMU orientation into image motion
 *
 * Idea: when the camera only ROTATES between two frames, every pixel moves by
 * the same 3x3 homography:
 *
 *        u_to  ~  K * R_to^T * R_from * K^-1 * u_from
 *
 *   K       camera intrinsics (focal length fx, fy and image centre cx, cy)
 *   R_from  camera orientation (camera -> world) in the earlier frame
 *   R_to    camera orientation in the later frame
 *
 * Camera orientation comes from the IMU:  R_wc = R_wb(q) * R_bc
 *   R_wb  IMU orientation from the ESKF (body -> world)
 *   R_bc  how the IMU is mounted on the camera (camera axes in IMU coordinates)
 *
 * Camera axes use the OpenCV convention: x = right, y = down, z = forward.
 *
 * Only ROTATION is compensated. Walking forward/sideways moves
 * near objects more than far ones and can't be removed without depth.
 * rotation moves pixels which adjust the bounding boxes
 */
#pragma once
#include <array>
#include <cmath>
#include <string>

#include "mat.h"

struct CameraModel {
    double fx = 0, fy = 0, cx = 0, cy = 0;
    bool   from_calibration = false;
};

// Rough intrinsics from the horizontal field of view (used until you calibrate).
inline CameraModel camera_from_hfov(int width, int height, double hfov_deg) {
    CameraModel K;
    K.fx = K.fy = (width / 2.0) / std::tan(hfov_deg * M_PI / 360.0);
    K.cx = width / 2.0;
    K.cy = height / 2.0;
    return K;
}

// Parse the IMU mounting, e.g. "+x,-z,+y" means:
//   camera right   (x) = IMU +x
//   camera down    (y) = IMU -z
//   camera forward (z) = IMU +y
// Returns false with a message if the spec is malformed or not a proper rotation.
inline bool parse_cam_axes(const std::string& spec, eskf::Mat3& R_bc, std::string& err) {
    std::array<std::string, 3> tok;
    size_t start = 0;
    for (int i = 0; i < 3; ++i) {
        size_t comma = spec.find(',', start);
        if (i < 2 && comma == std::string::npos) { err = "need 3 comma-separated axes like +x,+y,+z"; return false; }
        tok[i] = spec.substr(start, (i < 2 ? comma : spec.size()) - start);
        start = comma + 1;
    }
    R_bc = eskf::Mat3::Zero();
    for (int col = 0; col < 3; ++col) {
        const std::string& t = tok[col];
        if (t.size() != 2 || (t[0] != '+' && t[0] != '-') || t[1] < 'x' || t[1] > 'z') {
            err = "bad axis '" + t + "' (use +x, -y, +z ...)"; return false;
        }
        int row = t[1] - 'x';
        R_bc(row, col) = (t[0] == '+') ? 1.0 : -1.0;
    }
    // must be a proper rotation: det = +1 (right-handed)
    const eskf::Mat3& R = R_bc;
    double det = R(0,0)*(R(1,1)*R(2,2)-R(1,2)*R(2,1)) - R(0,1)*(R(1,0)*R(2,2)-R(1,2)*R(2,0))
               + R(0,2)*(R(1,0)*R(2,1)-R(1,1)*R(2,0));
    if (std::abs(det - 1.0) > 1e-9) {
        err = (std::abs(det + 1.0) < 1e-9) ? "axes are left-handed (flip the sign of one axis)"
                                           : "an IMU axis is used twice";
        return false;
    }
    return true;
}

// Camera orientation (camera -> world) from the IMU quaternion and the mounting.
inline eskf::Mat3 camera_orientation(const eskf::Quat& q_wb, const eskf::Mat3& R_bc) {
    return eskf::quat_to_rotmat(q_wb) * R_bc;
}

using Homography = std::array<float, 9>;   // row-major 3x3

// Pixel motion between two camera orientations (pure rotation).
inline Homography rotation_homography(const CameraModel& cam,
                                      const eskf::Mat3& R_wc_from,
                                      const eskf::Mat3& R_wc_to) {
    eskf::Mat3 K, Kinv;
    K(0,0) = cam.fx; K(0,2) = cam.cx; K(1,1) = cam.fy; K(1,2) = cam.cy; K(2,2) = 1;
    Kinv(0,0) = 1/cam.fx; Kinv(0,2) = -cam.cx/cam.fx;
    Kinv(1,1) = 1/cam.fy; Kinv(1,2) = -cam.cy/cam.fy; Kinv(2,2) = 1;
    eskf::Mat3 H = K * R_wc_to.T() * R_wc_from * Kinv;
    Homography out;
    for (int i = 0; i < 9; ++i) out[i] = (float)H.a[i];
    return out;
}

// Apply H to one pixel. Returns false if the point ends up behind the camera.
inline bool apply_homography(const Homography& H, float x, float y, float& ox, float& oy) {
    float w = H[6]*x + H[7]*y + H[8];
    if (w <= 1e-6f) return false;
    ox = (H[0]*x + H[1]*y + H[2]) / w;
    oy = (H[3]*x + H[4]*y + H[5]) / w;
    return true;
}
