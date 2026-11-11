# From ECE8880 f26 Lab 4 (scripts/), used unchanged by src/host/Makefile.
# Routed resource utilization of the kernel and each of its TAPA tasks.
# Usage: vivado -mode batch -source kernel_util.tcl -tclargs <dcp> <kernel cell> <report>
lassign $argv dcp kernel report
open_checkpoint $dcp
report_utilization -hierarchical -hierarchical_depth 3 -cells [get_cells $kernel] -file $report
