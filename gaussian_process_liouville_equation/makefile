CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -Ofast -march=native -fopenmp
LDFLAGS := ${CXXFLAGS} -Wl,-fuse-ld=gold -Wl,--no-as-needed
LDLIBS := -lnlopt -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lpthread
Objects := io.o pes.o mc.o kernel.o opt.o evolve.o main.o
PCH := stdafx.h.gch

.PHONY: all
all: gple.x

gple.x: ${Objects}
	${CXX} ${Objects} ${LDFLAGS} -o gple.x ${LDLIBS}
main.o: main.cpp evolve.h opt.h kernel.h mc.h pes.h io.h ${PCH}
	${CXX} -c main.cpp ${CXXFLAGS} -o main.o
evolve.o: evolve.cpp evolve.h opt.h kernel.h mc.h pes.h ${PCH}
	${CXX} -c evolve.cpp ${CXXFLAGS} -o evolve.o
opt.o: opt.cpp opt.h kernel.h mc.h io.h ${PCH}
	${CXX} -c opt.cpp ${CXXFLAGS} -o opt.o
kernel.o: kernel.cpp kernel.h ${PCH}
	${CXX} -c kernel.cpp ${CXXFLAGS} -o kernel.o
mc.o: mc.cpp mc.h io.h ${PCH}
	${CXX} -c mc.cpp ${CXXFLAGS} -o mc.o
pes.o: pes.cpp pes.h ${PCH}
	${CXX} -c pes.cpp ${CXXFLAGS} -o pes.o
io.o: io.cpp io.h ${PCH}
	${CXX} -c io.cpp ${CXXFLAGS} -o io.o
${PCH}: stdafx.h
	${CXX} stdafx.h ${CXXFLAGS}

.PHONY: clean
clean:
	\rm -f *.o *.gch

.PHONY: distclean
distclean: clean
	\rm -rf -- *log *out* *.png *.gif gp_test core.* *.txt

CLANG_FORMAT := $(shell grep alias\ clang-format= ~/.bash_aliases | awk -F"'" '{print $$2}')
.PHONY: format
format:
	for f in *.c* *.h*; \
	do \
		${CLANG_FORMAT} --style=file -i $${f}; \
	done
