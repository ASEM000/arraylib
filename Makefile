# Compiler and flags
CC = gcc
CFLAGS = -Wall -fPIC

# Target shared library
TARGET = arraylib/src/arraylib.so
SOURCES = arraylib/src/arraylib.c

# Include directories
INCLUDES = -I./arraylib/src

# Doxygen configuration
DOXY_FILE = arraylib/src/Doxyfile
DOXY_DIR = arraylib/src/docs

# Test directory and files
TEST_DIR = arraylib/src
TEST_SOURCES = $(wildcard $(TEST_DIR)/test*.c)
TEST_TARGETS = $(patsubst $(TEST_DIR)/%.c, %, $(TEST_SOURCES))

# Default target
all: shared docs

# Build the shared library
shared: $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -o $(TARGET) $(SOURCES)

# Generate documentation
docs:
	doxygen $(DOXY_FILE)

# Build and run all tests
tests: $(TEST_TARGETS)
	@for test in $(TEST_TARGETS); do \
		echo "running $$test..."; \
		./$$test || exit 1; \
	done

# Compile individual test files
%: $(TEST_DIR)/%.c shared
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(TARGET)

# Clean up
clean:
	rm -f $(TARGET) $(TEST_TARGETS)
	rm -rf $(DOXY_DIR)

.PHONY: all shared docs tests clean