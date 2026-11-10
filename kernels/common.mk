# Shared build rules for kernels/<name>/Makefile (same toolchain setup as the
# ECE 8880 labs). A kernel Makefile sets NAME (file prefix), TOP (kernel
# function) and optionally KERNEL_DEFS / TB_ARGS, then includes this file.
# Every generated file goes to build/kernels/<name>/ (B below); the kernel
# folder keeps only its sources.
XILINX_VERSION := 2024.2
XILINX_ROOT := /tools/Xilinx
export XILINX_VITIS := $(XILINX_ROOT)/Vitis/$(XILINX_VERSION)
export XILINX_HLS := $(XILINX_ROOT)/Vitis_HLS/$(XILINX_VERSION)
export XILINX_VIVADO := $(XILINX_ROOT)/Vivado/$(XILINX_VERSION)
export PLATFORM_REPO_PATHS := /opt/xilinx/platforms
export PATH := $(HOME)/.rapidstream-tapa/usr/bin:$(XILINX_VITIS)/bin:$(XILINX_HLS)/bin:$(XILINX_VIVADO)/bin:$(PATH)
export TAPA_CONCURRENCY ?= 8
SHELL := /bin/bash

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PLATFORM := xilinx_u55c_gen3x16_xdma_3_202210_1
CLOCK_PERIOD := 3.33
DEFS := -DGPT2_DEFAULT_MODEL_DIR=\"$(ROOT)/models\" -DGPT2_DEFAULT_DATA_DIR=\"$(ROOT)/data\"
GXX_FLAGS := -w -Wno-psabi -O2 -std=c++17 -I$(XILINX_VITIS)/include $(DEFS) $(KERNEL_DEFS)
LIB := -ltapa -lfrt -lglog -lgflags -lOpenCL
B := $(ROOT)/build/kernels/$(NAME)
TAPA := tapa --work-dir $(B)/work.out
REF_SRCS := $(ROOT)/src/cpu/ops.cpp $(ROOT)/src/common/weights.cpp
HDRS := $(wildcard *.h $(ROOT)/kernels/common/*.h $(ROOT)/src/common/*.h $(ROOT)/src/cpu/*.h) \
        $(ROOT)/data/golden_loader.h

.DEFAULT_GOAL := tb
.PHONY: tb swsim hls cosim hwemu clean cleanall

$(B)/kernel.o: kernel.cpp $(HDRS) Makefile
	@mkdir -p $(B)
	$(TAPA) g++ -- $(GXX_FLAGS) -c $< -o $@

$(B)/tb.o: tb.cpp $(HDRS) Makefile
	@mkdir -p $(B)
	$(TAPA) g++ -- $(GXX_FLAGS) -c $< -o $@

# CPU reference operators (src/cpu) that the testbench compares against
$(B)/ref.o: $(REF_SRCS) $(HDRS)
	@mkdir -p $(B)
	g++ -O2 -std=c++17 -w -c $(ROOT)/src/cpu/ops.cpp -o $(B)/ref_ops.o
	g++ -O2 -std=c++17 -w $(DEFS) -c $(ROOT)/src/common/weights.cpp -o $(B)/ref_weights.o
	ld -r $(B)/ref_ops.o $(B)/ref_weights.o -o $@

$(B)/tb: $(B)/kernel.o $(B)/tb.o $(B)/ref.o
	$(TAPA) g++ -- $(GXX_FLAGS) -o $@ $^ $(LIB)

tb: $(B)/tb

# software simulation: testbench vs the CPU reference
swsim: $(B)/tb
	$(B)/tb $(TB_ARGS)

# HLS synthesis -> $(B)/$(NAME).xo, reports in $(B)/work.out/report, log in $(B)/hls.log
$(B)/$(NAME).xo: kernel.cpp $(HDRS) Makefile
	@mkdir -p $(B)
	$(TAPA) compile -j 8 --top $(TOP) \
	--platform $(PLATFORM) \
	--clock-period $(CLOCK_PERIOD) \
	$(foreach d,$(KERNEL_DEFS),-c $(d)) \
	-f $< \
	-o $@ 2>&1 | tee $(B)/hls.log; test $${PIPESTATUS[0]} -eq 0

hls: $(B)/$(NAME).xo

# TAPA fast RTL cosimulation of the synthesized kernel (the labs' "hwemu").
cosim: $(B)/tb $(B)/$(NAME).xo
	env -u XILINX_XRT $(B)/tb --bitstream=$(B)/$(NAME).xo $(COSIM_ARGS)

hwemu: cosim

clean:
	rm -f $(B)/*.o $(B)/tb

cleanall:
	rm -rf $(B)
