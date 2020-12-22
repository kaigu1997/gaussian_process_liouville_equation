CXX := icpc
WARNINGFLAGS := -Wall -Wextra
CXXFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -fast -fopenmp -pch
LDFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -ipo -Ofast -xHost -Wl,-fuse-ld=gold -fopenmp
LDLIBS := -lpthread -lnlopt -lshogun
Objects := io.o gpr.o
PCH := stdafx.h
HEADERS := ${PCH} gpr.h io.h

.PHONY: all
all: evolve_gp_test.x screenshot_gp_test.x continue_gp_test.x

evolve_gp_test.x: ${Objects} main_evolve.o
	${CXX} ${Objects} main_evolve.o ${LDFLAGS} -o evolve_gp_test.x ${LDLIBS}
main_evolve.o: main_evolve.cpp ${HEADERS}
	${CXX} -c main_evolve.cpp ${CXXFLAGS} -o main_evolve.o

screenshot_gp_test.x: ${Objects} main_screenshot.o
	${CXX} ${Objects} main_screenshot.o ${LDFLAGS} -o screenshot_gp_test.x ${LDLIBS}
main_screenshot.o: main_screenshot.cpp ${HEADERS}
	${CXX} -c main_screenshot.cpp ${CXXFLAGS} -o main_screenshot.o

gpr.o: gpr.cpp ${HEADERS}
	${CXX} -c gpr.cpp ${CXXFLAGS} -o gpr.o
io.o: io.cpp io.h ${PCH}
	${CXX} -c io.cpp ${CXXFLAGS} -o io.o

continue_gp_test.x: continue_test.cpp
	${CXX} continue_test.cpp ${LDFLAGS} -o con_gp_test ${LDLIBS}

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
