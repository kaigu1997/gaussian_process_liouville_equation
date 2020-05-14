CXX := icpc
WARNINGFLAGS := -Wall -Wextra
CXXFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -fast -fopenmp
LDFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -ipo -Ofast -xHost -Wl,-fuse-ld=gold -fopenmp
LDLIBS := -lpthread -lgsl
Objects := rand.o gpr.o main.o
HeaderFile := rand.h gpr.h

.PHONY: all
all: gp_test 

gp_test: ${Objects}
	${CXX} ${Objects} ${LDFLAGS} -o gp_test ${LDLIBS}
main.o: main.cpp ${HeaderFile}
	${CXX} -c main.cpp ${CXXFLAGS} -o main.o
gpr.o: gpr.cpp ${HeaderFile}
	${CXX} -c gpr.cpp ${CXXFLAGS} -o gpr.o
rand.o: rand.cpp rand.h
	${CXX} -c rand.cpp ${CXXFLAGS} -o rand.o

.PHONY: clean
clean:
	\rm -f *.o

.PHONY: distclean
distclean: clean
	\rm -rf -- *log *out* input *.txt *.png *.gif dvr
