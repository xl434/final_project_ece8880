# Sources and flags for programs that run the whole model through the
# kernels (src/host/host_main, tests/*). Include after setting ROOT.
# Objects go to build/obj, programs to build/bin.
XILINX_VERSION := 2024.2
XILINX_ROOT := /tools/Xilinx
export XILINX_VITIS := $(XILINX_ROOT)/Vitis/$(XILINX_VERSION)
export XILINX_HLS := $(XILINX_ROOT)/Vitis_HLS/$(XILINX_VERSION)
export PATH := $(HOME)/.rapidstream-tapa/usr/bin:$(XILINX_VITIS)/bin:$(XILINX_HLS)/bin:$(PATH)
export TAPA_CONCURRENCY ?= 8

DEFS := -DGPT2_DEFAULT_MODEL_DIR=\"$(ROOT)/models\" -DGPT2_DEFAULT_DATA_DIR=\"$(ROOT)/data\"
GXX_FLAGS := -w -Wno-psabi -O2 -std=c++17 -I$(XILINX_VITIS)/include $(DEFS)
LIB := -ltapa -lfrt -lglog -lgflags -lOpenCL
KERNELS := gemv layernorm gelu residual attention lm_head
BUILD := $(ROOT)/build/obj
BIN := $(ROOT)/build/bin
TAPA := tapa --work-dir $(ROOT)/build/tapa.work
KERNEL_OBJS := $(foreach k,$(KERNELS),$(BUILD)/k_$(k).o)
HOST_OBJS := $(BUILD)/fpga_model.o $(BUILD)/ops.o $(BUILD)/weights.o $(BUILD)/tokenizer.o
ALL_HDRS := $(wildcard $(ROOT)/kernels/*/*.h $(ROOT)/src/*/*.h) $(ROOT)/data/golden_loader.h

$(BUILD)/k_%.o: $(ROOT)/kernels/%/kernel.cpp $(ALL_HDRS)
	@mkdir -p $(BUILD)
	$(TAPA) g++ -- $(GXX_FLAGS) -c $< -o $@

$(BUILD)/fpga_model.o: $(ROOT)/src/host/fpga_model.cpp $(ALL_HDRS)
	@mkdir -p $(BUILD)
	$(TAPA) g++ -- $(GXX_FLAGS) -c $< -o $@

$(BUILD)/ops.o: $(ROOT)/src/cpu/ops.cpp $(ALL_HDRS)
	@mkdir -p $(BUILD)
	g++ -O2 -std=c++17 -w -c $< -o $@

$(BUILD)/weights.o: $(ROOT)/src/common/weights.cpp $(ALL_HDRS)
	@mkdir -p $(BUILD)
	g++ -O2 -std=c++17 -w $(DEFS) -c $< -o $@

$(BUILD)/tokenizer.o: $(ROOT)/src/common/tokenizer.cpp $(ALL_HDRS)
	@mkdir -p $(BUILD)
	g++ -O2 -std=c++17 -w -c $< -o $@
