/* WP-G G7 workaround, recorded honestly rather than hidden.
 *
 * Vivado 2024.2 CRASHES (SIGABRT, "realloc(): invalid pointer") inside
 * libudev.so.1's udev_enumerate_scan_devices, called from libXil_lmgr11.so while
 * checking out the 'Synthesis' feature -- i.e. before it ever reaches the licence
 * server. The container has no /run/udev, so the scan walks a directory tree that
 * is not there. Stack in synth/vivlic/hs_err_pid23142.log.
 *
 * This LD_PRELOAD makes udev_enumerate_scan_devices a no-op returning success, so
 * the enumeration yields an empty device list instead of corrupting the heap.
 * The devices it enumerates are for dongle-based (node-locked USB) licences; this
 * box uses a FLOATING licence server (XILINXD_LICENSE_FILE=2100@...), which is
 * resolved over the network and does not need them.
 *
 * Everything else is passed through to the real libudev.
 */
#define _GNU_SOURCE
#include <stddef.h>
int udev_enumerate_scan_devices(void *e) { (void)e; return 0; }
int udev_enumerate_scan_subsystems(void *e) { (void)e; return 0; }
