/* WP-G G7: a NO-OP libudev.so.1 for Vivado's licence manager ONLY.
 *
 * Vivado 2024.2 and 2025.2 both SIGABRT/SIGSEGV inside the real libudev's
 * udev_enumerate_scan_devices, called from libXil_lmgr11.so while checking out the
 * 'Synthesis' feature (stacks in the hs_err_pid logs under synth/vivlic and
 * synth/vivlic25). The container has no
 * /run/udev. libXil_lmgr11 dlopen()s "libudev.so.1" and resolves the symbols on that
 * handle, so an LD_PRELOAD is bypassed -- the only way in is to be the libudev that
 * dlopen finds, i.e. to sit earlier on LD_LIBRARY_PATH.
 *
 * Every entry point is a no-op: enumerations succeed and return EMPTY lists, handles are
 * a single dummy object. The devices being enumerated are for DONGLE (node-locked USB)
 * licences; this box uses a FLOATING server licence (XILINXD_LICENSE_FILE=2100@...),
 * which is resolved over the network. If the checkout still fails, that is a real licence
 * answer and is reported as one.
 *
 * Used ONLY via LD_LIBRARY_PATH on the Vivado process. Generated, do not hand-edit.
 */
#include <stddef.h>
static char _dummy[64];

void *udev_device_get_action(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_current_tags_list_entry(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_devlinks_list_entry(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_devnode(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_devnum(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_devpath(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_devtype(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_driver(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_is_initialized(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_parent(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_parent_with_subsystem_devtype(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_properties_list_entry(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_property_value(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_seqnum(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_subsystem(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_sysattr_list_entry(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_sysattr_value(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_sysname(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_sysnum(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_syspath(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_tags_list_entry(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_udev(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_get_usec_since_initialized(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_has_current_tag(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_has_tag(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_new_from_device_id(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_new_from_devnum(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_new_from_environment(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_new_from_subsystem_sysname(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_new_from_syspath(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_device_ref(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
int udev_device_set_sysattr_value(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
void *udev_device_unref(void *a) { (void)a; return NULL; }
int udev_enumerate_add_match_is_initialized(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_match_parent(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_match_property(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_match_subsystem(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_match_sysattr(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_match_sysname(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_match_tag(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_nomatch_subsystem(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_nomatch_sysattr(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_add_syspath(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
void *udev_enumerate_get_list_entry(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_enumerate_get_udev(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_enumerate_new(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
void *udev_enumerate_ref(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
int udev_enumerate_scan_devices(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_enumerate_scan_subsystems(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
void *udev_enumerate_unref(void *a) { (void)a; return NULL; }
void *udev_get_log_priority(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_get_userdata(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_hwdb_get_properties_list_entry(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_hwdb_new(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
void *udev_hwdb_ref(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
void *udev_hwdb_unref(void *a) { (void)a; return NULL; }
void *udev_list_entry_get_by_name(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_list_entry_get_name(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_list_entry_get_next(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_list_entry_get_value(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
int udev_monitor_enable_receiving(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_monitor_filter_add_match_subsystem_devtype(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_monitor_filter_add_match_tag(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_monitor_filter_remove(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_monitor_filter_update(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
void *udev_monitor_get_fd(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_monitor_get_udev(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_monitor_new_from_netlink(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
void *udev_monitor_receive_device(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_monitor_ref(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
int udev_monitor_set_receive_buffer_size(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
void *udev_monitor_unref(void *a) { (void)a; return NULL; }
void *udev_new(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
void *udev_queue_flush(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_queue_get_fd(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_queue_get_kernel_seqnum(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
int udev_queue_get_queue_is_empty(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_queue_get_queued_list_entry(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
void *udev_queue_get_seqnum_is_finished(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_queue_get_seqnum_sequence_is_finished(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_queue_get_udev(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_queue_get_udev_is_active(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_queue_get_udev_seqnum(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_queue_new(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
void *udev_queue_ref(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
void *udev_queue_unref(void *a) { (void)a; return NULL; }
void *udev_ref(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return (void *)_dummy; }
int udev_set_log_fn(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
int udev_set_log_priority(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return 0; }
void *udev_set_userdata(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
void *udev_unref(void *a) { (void)a; return NULL; }
void *udev_util_encode_string(void *a, void *b, void *c) { (void)a;(void)b;(void)c; return NULL; }
