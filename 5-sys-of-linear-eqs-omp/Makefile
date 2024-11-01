CC=clang++
CFLAGS=-std=c++17 -O3 -I/usr/lib/gcc/x86_64-linux-gnu/11/include
LDFLAGS=-fopenmp=libiomp5
SOURCES=solver.cc
OBJECTS=$(SOURCES:.cc=.o)
EXECUTABLE=solver

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cc.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS)

cleanall:
	rm -f $(OBJECTS) $(EXECUTABLE)
