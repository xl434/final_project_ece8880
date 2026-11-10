ROOT := $(abspath ..)
include $(ROOT)/src/host/host.mk

.DEFAULT_GOAL := block_test
.PHONY: block_test block e2e all clean

$(BUILD)/block_test.o: block_test.cpp $(ALL_HDRS)
	@mkdir -p $(BUILD)
	$(TAPA) g++ -- $(GXX_FLAGS) -c $< -o $@

$(BIN)/block_test: $(BUILD)/block_test.o $(HOST_OBJS) $(KERNEL_OBJS)
	@mkdir -p $(BIN)
	$(TAPA) g++ -- $(GXX_FLAGS) -o $@ $^ $(LIB)

block_test: $(BIN)/block_test

block: $(BIN)/block_test
	$(BIN)/block_test

e2e:
	$(MAKE) -C $(ROOT)/src/host swsim

all: block e2e

clean:
	rm -f $(BIN)/block_test $(BUILD)/block_test.o
