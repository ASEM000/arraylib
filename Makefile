CC = gcc
CFLAGS = -Wall -fPIC
TARGET = arraylib/arraylib.so
SOURCES = arraylib/arraylib.c
INCLUDES = -I./arraylib

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -o $@ $^

clean:
	rm -f $(TARGET)