/*
 * imu_estimator.h — runs BNO055 reading live instead of csv + ESKF in a background thread.
 *
 *     ImuEstimator imu;
 *     imu.start();                       // calibrates (keep still 2 s), then runs
 *     ...  in the camera loop:
 *     ImuState s = imu.latest();         // thread-safe snapshot
 *     if (s.ready) { use s.roll_deg / s.pitch_deg / s.yaw_deg / s.q }
 *     ...
 *     imu.stop();
 *
 * The camera loop (~50 FPS) and IMU loop (~100 Hz) run independently;
 * latest() just copies the newest estimate under a mutex.
 */
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "bno055.h"
#include "eskf.h"

struct ImuState {
    bool   ready = false;           // false while calibrating
    double t = 0;                   // seconds since start
    double roll_deg = 0, pitch_deg = 0, yaw_deg = 0;
    eskf::Vec3 sigma3_deg;          // ±3σ per axis
    eskf::Vec3 bias_dps;            // current gyro bias estimate
    eskf::Quat q;                   // full orientation (body -> world)
    bool   accel_used = false;      // was the last accel update applied (not gated)?
    long   samples = 0;
};

struct ImuOptions {
    std::string dev = "/dev/i2c-1";
    int    addr = 0x28;
    double rate_hz = 100.0;
    std::string raw_log_path;   // if set: log raw data in imu_log.csv format (replayable offline)
};

class ImuEstimator {
public:
    using Options = ImuOptions;

    ~ImuEstimator() { stop(); }

    bool start(const Options& opt = Options(), const eskf::Config& cfg = eskf::Config()) {
        opt_ = opt; cfg_ = cfg;
        if (!dev_.open(opt.dev.c_str(), opt.addr) || !dev_.begin()) return false;
        running_ = true;
        worker_ = std::thread(&ImuEstimator::run, this);
        return true;
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
    }

    ImuState latest() const { std::lock_guard<std::mutex> lk(mtx_); return state_; }
    bool ok() const { return !failed_; }

private:
    using Clock = std::chrono::steady_clock;

    Options          opt_;
    eskf::Config     cfg_;
    BNO055           dev_;
    std::thread      worker_;
    std::atomic<bool> running_{false};
    std::atomic<bool> failed_{false};
    mutable std::mutex mtx_;
    ImuState         state_;

    void run() {
        std::ofstream log;
        if (!opt_.raw_log_path.empty()) {
            log.open(opt_.raw_log_path);
            log << "t_ms,ax,ay,az,gx,gy,gz\n";
        }

        const auto period = std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(1.0 / opt_.rate_hz));
        const auto t_start = Clock::now();
        auto next_tick = t_start;

        std::vector<eskf::Vec3> cal_g, cal_a;
        eskf::ESKF filt;
        bool calibrated = false;
        eskf::Vec3 prev_gyro;
        double prev_t = 0;
        long n = 0, read_errors = 0;

        while (running_) {
            next_tick += period;
            int16_t acc[3], gyr[3];
            if (!dev_.read_raw(acc, gyr)) {
                if (++read_errors > 50) { std::fprintf(stderr, "IMU: too many I2C read errors, stopping\n"); failed_ = true; break; }
                std::this_thread::sleep_until(next_tick);
                continue;
            }
            const double t = std::chrono::duration<double>(Clock::now() - t_start).count();
            if (log.is_open())
                log << (long)(t * 1000) << ',' << acc[0] << ',' << acc[1] << ',' << acc[2]
                    << ',' << gyr[0] << ',' << gyr[1] << ',' << gyr[2] << '\n';

            eskf::Vec3 g = eskf::raw_gyro_to_rads(eskf::vec3(gyr[0], gyr[1], gyr[2]), cfg_);
            eskf::Vec3 a = eskf::vec3(acc[0], acc[1], acc[2]);
            ++n;

            if (!calibrated) {
                // ── Startup calibration: sensor must be still ──
                cal_g.push_back(g); cal_a.push_back(a);
                if (t >= cfg_.cal_sec && cal_g.size() >= 20) {
                    eskf::Calibration cal = eskf::calibrate(cal_g, cal_a, cfg_);
                    filt.init(cal, cfg_, true);
                    calibrated = true;
                    std::printf("IMU calibrated: bias [%.4f %.4f %.4f] deg/s  sigma_g %.2e rad/s  tilt roll %.2f pitch %.2f deg\n",
                        cal.gyro_bias[0]*180/M_PI, cal.gyro_bias[1]*180/M_PI, cal.gyro_bias[2]*180/M_PI,
                        cal.sigma_g, cal.roll0*180/M_PI, cal.pitch0*180/M_PI);
                }
            } else {
                // ── Filter: predict with previous gyro over dt, then accel update ──
                double dt = t - prev_t;
                bool used = false;
                if (dt > 0 && dt < 0.5) {             // skip absurd gaps (e.g. after a stall)
                    filt.predict(prev_gyro, dt);
                    used = filt.update_accel(a);
                }
                eskf::Vec3 e = filt.euler();
                std::lock_guard<std::mutex> lk(mtx_);
                state_.ready = true;
                state_.t = t;
                state_.roll_deg  = e[0] * 180 / M_PI;
                state_.pitch_deg = e[1] * 180 / M_PI;
                state_.yaw_deg   = e[2] * 180 / M_PI;
                state_.sigma3_deg = filt.sigma3_deg();
                state_.bias_dps   = filt.b * (180 / M_PI);
                state_.q = filt.q;
                state_.accel_used = used;
                state_.samples = n;
            }
            prev_gyro = g;
            prev_t = t;
            std::this_thread::sleep_until(next_tick);
        }
    }
};
