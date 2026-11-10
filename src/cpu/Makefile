ROOT := $(abspath ../..)
CXX := g++
CXXFLAGS := -O3 -march=native -fopenmp -std=c++17 -Wall
DEFS := -DGPT2_DEFAULT_MODEL_DIR=\"$(ROOT)/models\" \
        -DGPT2_DEFAULT_DATA_DIR=\"$(ROOT)/data\" \
        -DGPT2_DEFAULT_RESULTS_DIR=\"$(ROOT)/results\" \
        -DGPT2_CXXFLAGS="\"$(CXXFLAGS)\""
SRCS := cpu_main.cpp gpt2_cpu.cpp ops.cpp ../common/weights.cpp ../common/tokenizer.cpp
BIN := $(ROOT)/build/bin
HDRS := $(wildcard *.h ../common/*.h ../../data/golden_loader.h)

.DEFAULT_GOAL := cpu_main
.PHONY: cpu_main generate verify profile clean

$(BIN)/cpu_main: $(SRCS) $(HDRS) Makefile
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ $(SRCS)

cpu_main: $(BIN)/cpu_main

generate: $(BIN)/cpu_main
	$(BIN)/cpu_main generate

verify: $(BIN)/cpu_main
	$(BIN)/cpu_main verify

profile: $(BIN)/cpu_main
	$(BIN)/cpu_main profile --threads 1,8 --ctx 128,512,1024 --reps 3

clean:
	rm -f $(BIN)/cpu_main
