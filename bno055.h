/*
 * bno055.h — minimal BNO055 reader over Linux I2C (/dev/i2c-N), no libraries.
 *
 * Puts the chip in ACCGYRO mode: raw accelerometer + gyroscope, with the
 * BNO055's own onboard fusion OFF — my ESKF does the fusion.
 *
 * Units (default UNIT_SEL):
 *   accel: 1 m/s^2 = 100 LSB    (only the direction is used by the filter)
 *   gyro : 1 deg/s = 16 LSB     (matches Config::lsb_per_dps)
 *
 * Wiring that works on your Orin Nano: SDA/SCL on header pins 27/28 -> bus 1,
 * address 0x28  (confirmed: i2cget -y 1 0x28 0x00 -> 0xa0).
 */
#pragma once
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <thread>
#include <chrono>

class BNO055 {
public:
    // Registers (page 0)
    static constexpr uint8_t REG_CHIP_ID  = 0x00;   // reads 0xA0
    static constexpr uint8_t REG_PAGE_ID  = 0x07;
    static constexpr uint8_t REG_ACC_X_LSB = 0x08;  // accel 0x08..0x0D, mag 0x0E..0x13, gyro 0x14..0x19
    static constexpr uint8_t REG_UNIT_SEL = 0x3B;
    static constexpr uint8_t REG_OPR_MODE = 0x3D;
    static constexpr uint8_t REG_PWR_MODE = 0x3E;

    static constexpr uint8_t MODE_CONFIG  = 0x00;
    static constexpr uint8_t MODE_ACCGYRO = 0x05;

    ~BNO055() { if (fd_ >= 0) ::close(fd_); }

    bool open(const char* dev = "/dev/i2c-1", int addr = 0x28) {
        fd_ = ::open(dev, O_RDWR);
        if (fd_ < 0) { std::perror("open i2c device"); return false; }
        if (ioctl(fd_, I2C_SLAVE, addr) < 0) { std::perror("ioctl I2C_SLAVE"); return false; }
        return true;
    }

    // Configure for raw accel+gyro streaming. Returns false if the chip isn't there.
    bool begin() {
        uint8_t id = 0;
        if (!read_regs(REG_CHIP_ID, &id, 1) || id != 0xA0) {
            std::fprintf(stderr, "BNO055 not found (chip id 0x%02X, expected 0xA0)\n", id);
            return false;
        }
        write_reg(REG_OPR_MODE, MODE_CONFIG);  sleep_ms(25);   // must be in CONFIG to change settings
        write_reg(REG_PAGE_ID, 0x00);
        write_reg(REG_PWR_MODE, 0x00);         sleep_ms(10);   // normal power
        write_reg(REG_UNIT_SEL, 0x00);                         // m/s^2, deg/s
        write_reg(REG_OPR_MODE, MODE_ACCGYRO); sleep_ms(25);   // raw accel + gyro, no onboard fusion
        return true;
    }

    // One burst read of accel + mag + gyro (18 bytes). Values are raw int16 counts.
    bool read_raw(int16_t acc[3], int16_t gyr[3]) {
        uint8_t buf[18];
        if (!read_regs(REG_ACC_X_LSB, buf, 18)) return false;
        for (int i = 0; i < 3; ++i) {
            acc[i] = (int16_t)(buf[2*i]      | (buf[2*i + 1]  << 8));
            gyr[i] = (int16_t)(buf[12 + 2*i] | (buf[13 + 2*i] << 8));
        }
        return true;
    }

private:
    int fd_ = -1;

    static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

    bool write_reg(uint8_t reg, uint8_t val) {
        uint8_t b[2] = {reg, val};
        return ::write(fd_, b, 2) == 2;
    }
    bool read_regs(uint8_t reg, uint8_t* out, int n) {
        if (::write(fd_, &reg, 1) != 1) return false;
        return ::read(fd_, out, n) == n;
    }
};
