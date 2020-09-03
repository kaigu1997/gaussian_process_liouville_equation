CXX := icpc
WARNINGFLAGS := -Wall -Wextra
CXXFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -fast -fopenmp
LDFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -ipo -Ofast -xHost -Wl,-fuse-ld=gold -fopenmp
LDLIBS := -lpthread
Objects := general.o pes.o main.o
HeaderFile := general.h pes.h

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

.PHONY: clean
clean:
	\rm -f *.o

.PHONY: distclean
distclean: clean
	\rm -rf -- *log *out* input *.txt *.png *.gif dvr

.PHONY: format
format:
	for f in *.c* *.h*; \
	do \
		clang-format --style=file -i $${f}; \
	done
