#
SHELL=/bin/sh
OBJECTS = gp.o
OBJECTS-Multi = gpMultiDimensional.o

INCLUDE = -I /Users/jmschofi/eigen/ -I /Users/jmschofi/gpr
cc=g++ -std=c++11
#flags=-g -c $(INCLUDE)
flags = -O3 -c $(INCLUDE)
links = /Users/jmschofi/gpr/libgp.a -lgsl -lgslcblas 

all: gp gpMulti

gp: $(OBJECTS)
	$(cc) $(OBJECTS) -o ../gp $(links)
gpMulti: $(OBJECTS-Multi)
	$(cc) $(OBJECTS-Multi) -o ../gpMulti $(links)


gp.o: gp.cxx
	$(cc) $(flags) gp.cxx

gpMultiDimensional.o: gpMultiDimensional.cxx
	$(cc) $(flags) gpMultiDimensional.cxx


clean:
	\rm -f *.o ../gp ../gpMulti

