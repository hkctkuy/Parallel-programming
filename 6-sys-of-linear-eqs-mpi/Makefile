CXX=mpic++
OMPI_CC=clang
OMPI_CXX=clang++
CXXFLAGS=-fopenmp=libiomp5 -std=c++17 -O3
LDFLAGS=-fopenmp=libiomp5
SOURCES=solver.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=solver

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	OMPI_CC=$(OMPI_CC) OMPI_CXX=$(OMPI_CXX) $(CXX) $(LDFLAGS) $(OBJECTS) -o $@

.cc.o:
	OMPI_CC=$(OMPI_CC) OMPI_CXX=$(OMPI_CXX) $(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS)

cleanall: clean
	rm -f $(EXECUTABLE)
