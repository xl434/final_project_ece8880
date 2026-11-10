ROOT := $(abspath ../..)
include host.mk

.DEFAULT_GOAL := host_main
.PHONY: host_main swsim plan clean

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

clean:
	rm -f $(BIN)/host_main $(BUILD)/host_main.o
