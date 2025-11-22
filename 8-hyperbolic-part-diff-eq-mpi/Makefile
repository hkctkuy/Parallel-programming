CXX			= mpic++
OMPI_CC		= clang
OMPI_CXX	= clang++
OMPFLAGS	= -fopenmp
SANFLAGS 	= -fsanitize=address,integer,bounds,null,undefined,float-divide-by-zero
CXXFLAGS	= $(OMPFLAGS) -std=c++20
LDFLAGS 	= $(OMPFLAGS)

SOURCES     = solver.cc
OBJECTS     = $(SOURCES:.cc=.o)
EXECUTABLE  = solver

BUILD_TYPE ?= Release

ifeq ($(BUILD_TYPE), Debug)
  CXXFLAGS += $(SANFLAGS) -O0 -g -DDEBUG
  LDFLAGS  += $(SANFLAGS)
else
  CXXFLAGS += -O3
endif

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	OMPI_CC=$(OMPI_CC) OMPI_CXX=$(OMPI_CXX) $(CXX) $(LDFLAGS) $(OBJECTS) -o $@

.cc.o:
	OMPI_CC=$(OMPI_CC) OMPI_CXX=$(OMPI_CXX) $(CXX) $(CXXFLAGS) -c $< -o $@

debug: cleanall
	$(MAKE) BUILD_TYPE=Debug

release: cleanall
	$(MAKE) BUILD_TYPE=Release

clean:
	rm -f $(OBJECTS)

cleanall: clean
	rm -f $(EXECUTABLE)

.PHONY: all clean cleanall debug release
