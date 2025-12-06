# Env
ARCH 	  	   ?= sm_60
BUILD_TYPE	   ?= Release

# Compilers
CC				= clang
CXX				= clang++
MPI_CC			= mpicc
MPI_CXX			= mpic++
MPI_HOST		= OMPI_CC=$(CC) OMPI_CXX=$(CXX)
HOST_COMP 	   ?= $(MPI_CXX)
NVCC			= nvcc

# Flags
STD				= -std=c++20
CUDAFLAGS		= -lcudart
OMPFLAGS		= -fopenmp
SANFLAGS		= -fsanitize=address,integer,bounds,null,undefined,float-divide-by-zero
CXXFLAGS		= $(STD) -MMD -MP $(OMPFLAGS) -Wno-macro-redefined

ifeq ($(BUILD_TYPE), Debug)
  CXXFLAGS     += $(SANFLAGS) -O0 -g -DDEBUG
else
  CXXFLAGS     += -O3
endif

NVCCFLAGS		= -arch=$(ARCH) -ccbin=$(HOST_COMP) $(STD) -Xcompiler "$(CXXFLAGS)"

# Files
CXX_SOURCES		= main.cc
CUDA_SOURCES   	= kernel.cu util.cu
OBJECTS    		= $(CXX_SOURCES:.cc=.o) $(CUDA_SOURCES:.cu=.o)
DEPENDS    		= $(CXX_SOURCES:.cc=.d)
EXECUTABLE 		= solver


all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(MPI_HOST) $(NVCC) $(NVCCFLAGS) $(OBJECTS) -o $@ $(CUDAFLAGS)

%.o: %.cc
	$(MPI_HOST) $(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu
	$(MPI_HOST) $(NVCC) $(NVCCFLAGS) -c $< -o $@

-include $(DEPENDS)

debug: cleanall
	$(MAKE) BUILD_TYPE=Debug

release: cleanall
	$(MAKE) BUILD_TYPE=Release

clean:
	rm -f $(OBJECTS) $(DEPENDS) $(CUDA_SOURCES:.cu=.d)

cleanall: clean
	rm -f $(EXECUTABLE)

.PHONY: all clean cleanall debug release
