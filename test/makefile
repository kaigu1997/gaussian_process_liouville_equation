CXX := icpc
WARNINGFLAGS := -Wall -Wextra
CXXFLAGS := ${WARNINGFLAGS} -mkl -std=c++14 -fast -fopenmp
LDFLAGS := ${WARNINGFLAGS} -mkl -std=c++14 -ipo -Ofast -xHost -Wl,-fuse-ld=gold -fopenmp
LDLIBS := -lpthread -lshogun
Objects := main.o

.PHONY: all
all: gp_test 

gp_test: ${Objects}
	${CXX} ${Objects} ${LDFLAGS} -o gp_test ${LDLIBS}
main.o: main.cpp
	${CXX} -c main.cpp ${CXXFLAGS} -o main.o

.PHONY: clean
clean:
	\rm -f *.o

.PHONY: distclean
distclean: clean
	\rm -rf -- *log *out* input *.txt *.png *.gif gp_test
