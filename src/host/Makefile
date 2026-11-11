ROOT := $(abspath ../..)
include host.mk

.DEFAULT_GOAL := host_main
.PHONY: host_main swsim plan xclbin_hw hw_util clean

$(BUILD)/host_main.o: host_main.cpp $(ALL_HDRS)
	@mkdir -p $(BUILD)
	$(TAPA) g++ -- $(GXX_FLAGS) -c $< -o $@

$(BIN)/host_main: $(BUILD)/host_main.o $(HOST_OBJS) $(KERNEL_OBJS)
	@mkdir -p $(BIN)
	$(TAPA) g++ -- $(GXX_FLAGS) -o $@ $^ $(LIB)

host_main: $(BIN)/host_main

# Full model, all operators in TAPA software simulation, checked against golden.
swsim: $(BIN)/host_main
	$(BIN)/host_main --check

plan: $(BIN)/host_main
	$(BIN)/host_main --plan

# Full hardware build of the kernels with the HBM plan (link_config.ini):
# v++ link + Vivado synthesis, placement and routing. No board needed; gives
# routed utilization and timing. Takes many hours (Lab 4: 8-20 h for one small
# kernel). The two workarounds are the ones from ECE8880 f26 Lab 4
# (src/host/vivado/): a stub libudev.so.1, loaded for Vivado only, because
# Vivado's license check crashes in this workspace's libudev; and a fix for
# Vitis 2024.2's clock-wizard call when a kernel frequency is given.
PLATFORM := xilinx_u55c_gen3x16_xdma_3_202210_1
CLOCK_PERIOD := 3.33
KERNEL_FREQ := $(shell awk 'BEGIN { printf "%.0f", 1000 / $(CLOCK_PERIOD) }')
export XILINX_VIVADO := $(XILINX_ROOT)/Vivado/$(XILINX_VERSION)
export PATH := $(XILINX_VIVADO)/bin:$(PATH)
# The U55C HBM interconnect (HMSS) has a limited number of AXI connections and
# every mmap argument uses one, so the standalone residual kernel (used only in
# the unfused test configuration) is left out of the hardware build.
HW_KERNELS := $(filter-out residual,$(KERNELS))
XOS := $(foreach k,$(HW_KERNELS),$(ROOT)/build/kernels/$(k)/$(k).xo)
HW := $(ROOT)/build/hw
VIV := $(CURDIR)/vivado
VIVADO_ENV := env LD_LIBRARY_PATH=$(HW)/compat:$(XILINX_VIVADO)/lib/lnx64.o:$$LD_LIBRARY_PATH
IMPL_RUN := $(HW)/_x/link/vivado/vpl/prj/prj.runs/impl_1
FINAL_DCP := $(IMPL_RUN)/level0_wrapper_postroute_physopt.dcp

$(HW)/link_hw.ini: link_config.ini
	@mkdir -p $(HW)
	grep -v 'ResidualKernel' $< > $@

$(HW)/compat/libudev.so.1: $(VIV)/udev_stub.c
	@mkdir -p $(HW)/compat
	gcc -shared -fPIC -o $@ $<

$(HW)/gpt2.hw.xclbin: $(XOS) $(HW)/link_hw.ini $(HW)/compat/libudev.so.1 $(VIV)/clkwiz_compat.tcl
	cd $(HW) && $(VIVADO_ENV) v++ --link --target hw --platform $(PLATFORM) \
	--config $(HW)/link_hw.ini \
	--kernel_frequency $(KERNEL_FREQ) \
	--vivado.prop run.impl_1.STEPS.INIT_DESIGN.TCL.PRE=$(VIV)/clkwiz_compat.tcl \
	--vivado.prop run.impl_1.STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED=true \
	--save-temps --temp_dir $(HW)/_x --log_dir $(HW)/logs --report_dir $(HW)/reports \
	-o $@ $(XOS) 2>&1 | tee $(HW)/vpp_link.log; test $${PIPESTATUS[0]} -eq 0

xclbin_hw: $(HW)/gpt2.hw.xclbin

# Routed utilization of each kernel and its TAPA tasks (after xclbin_hw).
hw_util: $(HW)/compat/libudev.so.1
	for k in GemvKernel LmHeadKernel AttentionKernel LayerNormKernel GeluKernel; do \
	  $(VIVADO_ENV) vivado -mode batch -nojournal -nolog -notrace -source $(VIV)/kernel_util.tcl \
	  -tclargs $(FINAL_DCP) level0_i/ulp/$${k}_1 $(HW)/util_$${k}.rpt || exit 1; done

clean:
	rm -f $(BIN)/host_main $(BUILD)/host_main.o
