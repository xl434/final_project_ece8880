// From ECE8880 f26 Lab 4 (scripts/), used unchanged by src/host/Makefile xclbin_hw.
// Some workspaces have no udev. Vivado's license check scans devices
// through libudev and crashes there; this stub reports no udev instead.
// The Lab 4 Makefiles build it into compat/libudev.so.1 for Vivado only.
struct udev;
struct udev *udev_new(void) { return 0; }
