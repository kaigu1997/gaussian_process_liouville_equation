CXX := icpc
WARNINGFLAGS := -Wall -Wextra -Wconversion -Wshadow -Werror
CXXFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -fast -g3 -fopenmp
LDFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -ipo -Ofast -xHost -Wl,-fuse-ld=gold -g3 -fopenmp
LDLIBS := -lpthread
Objects := matrix.o general.o pes.o main.o
HeaderFile := matrix.h general.h pes.h

.PHONY: all
all: dvr 

dvr: ${Objects}
	${CXX} ${Objects} ${LDFLAGS} -o dvr ${LDLIBS}
main.o: main.cpp ${HeaderFile}
	${CXX} -c main.cpp ${CXXFLAGS} -o main.o
pes.o: pes.cpp ${HeaderFile}
	${CXX} -c pes.cpp ${CXXFLAGS} -o pes.o
general.o: general.cpp ${HeaderFile}
	${CXX} -c general.cpp ${CXXFLAGS} -o general.o
matrix.o: matrix.cpp matrix.h
	${CXX} -c matrix.cpp ${CXXFLAGS} -o matrix.o

.PHONY: clean
clean:
	\rm -f *.o

.PHONY: distclean
distclean: clean
	\rm -rf -- *log *out* input *.txt *.png *.gif dvr
