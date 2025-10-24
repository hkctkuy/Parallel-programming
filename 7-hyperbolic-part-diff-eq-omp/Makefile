CC       = clang++
OMPFLAGS = -fopenmp=libiomp5
CFLAGS	 = $(OMPFLAGS) -std=c++20 -g -I/usr/lib/gcc/x86_64-linux-gnu/11/include
LDFLAGS  = $(OMPFLAGS)

SOURCES     = solver.cc
OBJECTS     = $(SOURCES:.cc=.o)
EXECUTABLE  = solver

BUILD_TYPE ?= Release

ifeq ($(BUILD_TYPE), Debug)
  CFLAGS += -O0 -DDEBUG
else
  CFLAGS += -O3
endif

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cc.o:
	$(CC) $(CFLAGS) -c $< -o $@

debug:
	$(MAKE) BUILD_TYPE=Debug

release:
	$(MAKE) BUILD_TYPE=Release

clean:
	rm -f $(OBJECTS)

cleanall:
	rm -f $(OBJECTS) $(EXECUTABLE)

.PHONY: all clean cleanall debug release
