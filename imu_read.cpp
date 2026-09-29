#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cstdint>
#include <cstdio>

int main() {
    int fd = open("/dev/i2c-1", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    if (ioctl(fd, I2C_SLAVE, 0x28) < 0) { perror("ioctl"); return 1; }

    uint8_t mode[2] = {0x3D, 0x0C};  // NDOF mode
    if (write(fd, mode, 2) != 2) { perror("mode"); return 1; }
    usleep(500000);

    auto read16 = [&](uint8_t reg) -> int16_t {
        write(fd, &reg, 1);
        uint8_t buf[2];
        if (read(fd, buf, 2) != 2) { perror("read"); return 0; }
        return (int16_t)((buf[1] << 8) | buf[0]);
    };

    for (int i = 0; i < 5; i++) {
        int16_t ax = read16(0x08), ay = read16(0x0A), az = read16(0x0C);
        int16_t gx = read16(0x14), gy = read16(0x16), gz = read16(0x18);
        printf("accel %d %d %d  gyro %d %d %d\n", ax, ay, az, gx, gy, gz);
        usleep(500000);
    }
    close(fd);
    return 0;
}
