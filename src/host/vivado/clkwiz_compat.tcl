# From ECE8880 f26 Lab 4 (scripts/), used unchanged by src/host/Makefile.
# Vitis 2024.2 fix for v++ --kernel_frequency on the U55C, used by the Lab 4
# Makefiles as the impl_1 INIT_DESIGN pre-hook.
#
# Vitis's ocl_util.tcl passes the project name where the installed
# clock-wizard API expects the part name. Vivado then logs
# "Invalid part string Project" and constrains the kernel clock to the
# 100 MHz MMCM input instead of the requested frequency. These four helpers
# are the originals with the part name passed instead.
namespace eval ::ocl_util {
  proc initialize_clkwiz_debug {is_versal partinfo} {
    load librdi_iptasks[info sharedlibextension]
    if {$is_versal} {
      Ip_Clocking_Init $partinfo
    } else {
      Init_Clkwiz $partinfo test1 $partinfo
    }
  }

  proc uninitialize_clkwiz_debug {is_versal partinfo} {
    if {$is_versal} {
      Ip_Clocking_Uninit $partinfo
    } else {
      UnInit_Clkwiz [get_property PART [current_project]] test1
    }
  }

  proc get_clkwiz_prop {prop} {
    return [GetClkwizProperty [get_property PART [current_project]] test1 $prop]
  }

  proc set_clkwiz_prop {clock_freq_orig clock_freq} {
    set part [get_property PART [current_project]]
    SetClkwizProperty $part test1 UseFinePS true
    GetClosestSolution $part test1 $clock_freq 0 50 $clock_freq_orig 0 1 false false false false false false
    puts "ECE8880: kernel clock requested $clock_freq MHz, MMCM M=[get_clkwiz_prop ChosenM] D=[get_clkwiz_prop ChosenD] O=[get_clkwiz_prop ChosenDiv0]"
  }
}
puts "ECE8880: clock-wizard fix loaded"
