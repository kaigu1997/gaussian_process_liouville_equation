CXX := g++
CXXFLAGS := -std=c++17 -Ofast -march=native -fopenmp
LDFLAGS := -std=c++17 -Ofast -march=native -Wl,-fuse-ld=gold -fopenmp
LDLIBS := -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lpthread -lnlopt -lshogun
Objects := io.o pes.o mc.o evolve.o main.o
PCH := stdafx.h.gch

.PHONY: all
all: gple.x

gple.x: ${Objects}
	${CXX} ${Objects} ${LDFLAGS} -o gple.x ${LDLIBS}
main.o: main.cpp evolve.h mc.h pes.h io.h ${PCH}
	${CXX} -c main.cpp ${CXXFLAGS} -o main.o
evolve.o: evolve.cpp evolve.h pes.h ${PCH}
	${CXX} -c evolve.cpp ${CXXFLAGS} -o evolve.o
mc.o: mc.cpp mc.h ${PCH}
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
	\rm -rf -- *log *out* input *.png *.gif gp_test core.* *.txt

CLANG_FORMAT := /home/j/jmschofi/kaigu/.vscode-server/extensions/ms-vscode.cpptools-1.1.3/LLVM/bin/clang-format
.PHONY: format
format:
	for f in *.c* *.h*; \
	do \
		${CLANG_FORMAT} --style=file -i $${f}; \
	done
