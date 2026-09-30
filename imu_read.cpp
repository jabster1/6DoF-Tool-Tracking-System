//bias calculation for sensor fusion and es-ekf step taken from csv - future implementation will gain bias numbers upon every startup
//gx_bias=0.228 gy_bias=0.075 gz_bias=-0.045
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cstdint>
#include <cstdio>
#include <chrono> // timestamps monotonic
#include <fstream> //std::ofstream


int main() {
    int fd = open("/dev/i2c-1", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    if (ioctl(fd, I2C_SLAVE, 0x28) < 0) { perror("ioctl"); return 1; }

    uint8_t mode[2] = {0x3D, 0x0C};  // NDOF mode
    if (write(fd, mode, 2) != 2) { perror("mode"); return 1; }
    usleep(500000);

    //create csv output file for imu output readings
    std::ofstream csv("imu_log.csv");
    csv << "t_ms,ax,ay,az,gx,gy,gz\n";
    auto t0 = std::chrono::steady_clock::now();

    auto read16 = [&](uint8_t reg) -> int16_t {
        write(fd, &reg, 1);
        uint8_t buf[2];
        if (read(fd, buf, 2) != 2) { perror("read"); return 0; }
        return (int16_t)((buf[1] << 8) | buf[0]);
    };

    //1000 samples 10 ms apart gives us 10 seconds of data
    for (int i = 0; i < 1000; i++) {
        // force resisting gravity, if z=937 then 937/100 = 9.37 m/s^2
        int16_t ax = read16(0x08), ay = read16(0x0A), az = read16(0x0C);
        //angular velocity - how fast sensor is rotating (raw/16)
        int16_t gx = read16(0x14), gy = read16(0x16), gz = read16(0x18);
        auto t = std::chrono::steady_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t - t0).count();
        csv << ms << "," << ax << "," << ay << "," << az << "," << gx << "," << gy << "," << gz << "\n";
        //printf("accel %d %d %d  gyro %d %d %d\n", ax, ay, az, gx, gy, gz);
        usleep(10000);
    }
    close(fd);
    csv.close();
    return 0;
}
