CXX := icpc
WARNINGFLAGS := -Wall -Wextra -Wconversion -Wshadow
CXXFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -fast -g3 -fopenmp
LDFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -ipo -Ofast -xHost -Wl,-fuse-ld=gold -g3 -fopenmp
LDLIBS := -lpthread
Objects := matrix.o general.o pes.o main.o
HeaderFile := matrix.h general.h pes.h

.PHONY: all
all: mqcl

mqcl: ${Objects}
	${CXX} ${LDFLAGS} ${Objects} -o mqcl ${LDLIBS}
main.o: main.cpp ${HeaderFile}
	${CXX} ${CXXFLAGS} -c main.cpp -o main.o
pes.o: pes.cpp ${HeaderFile}
	${CXX} ${CXXFLAGS} -c pes.cpp -o pes.o
general.o: general.cpp ${HeaderFile}
	${CXX} ${CXXFLAGS} -c general.cpp -o general.o
matrix.o: matrix.cpp matrix.h
	${CXX} ${CXXFLAGS} -c matrix.cpp -o matrix.o

.PHONY: clean
clean:
	-\rm *.o

.PHONY: distclean
distclean:
	\rm -rf -- *log *out* core.* *.txt *.png *.gif *.o mqcl
