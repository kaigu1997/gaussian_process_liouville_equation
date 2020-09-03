CXX := icpc
WARNINGFLAGS := -Wall -Wextra
CXXFLAGS := ${WARNINGFLAGS} -mkl -std=c++14 -fast -fopenmp -pch
LDFLAGS := ${WARNINGFLAGS} -mkl -std=c++14 -ipo -Ofast -xHost -Wl,-fuse-ld=gold -fopenmp
LDLIBS := -lpthread -lgsl
Objects := io.o gpr.o main.o
PCH := stdafx.h

.PHONY: all
all: gp_test 

gp_test: ${Objects}
	${CXX} ${Objects} ${LDFLAGS} -o gp_test ${LDLIBS}
main.o: main.cpp gpr.h io.h ${PCH}
	${CXX} -c main.cpp ${CXXFLAGS} -o main.o
gpr.o: gpr.cpp gpr.h io.h ${PCH}
	${CXX} -c gpr.cpp ${CXXFLAGS} -o gpr.o
io.o: io.cpp io.h ${PCH}
	${CXX} -c io.cpp ${CXXFLAGS} -o io.o

.PHONY: clean
clean:
	\rm -f *.o *.pchi

.PHONY: distclean
distclean: clean
	find *.txt | egrep -v "(^x|^p|^t)" | xargs rm
	\rm -rf -- *log *out* input *.png *.gif gp_test core.*

.PHONY: format
format:
	for f in *.c* *.h*; \
	do \
		clang-format --style=file -i $${f}; \
	done
