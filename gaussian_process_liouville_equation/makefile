CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++20 -O3 -march=native
LDFLAGS := $(CXXFLAGS) -Wl,-fuse-ld=gold -Wl,--no-as-needed
LDLIBS := -lspdlog -lnlopt -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core -ltbb -lpthread
sources := $(shell ls *.cpp)
objects := $(sources:.cpp=.o)
depends := $(sources:.cpp=.d)

PCH := stdafx.h.gch

.PHONY: all
all: $(depends) gple.x

# add .d to depends as well. If the header is updated, .d also need updating
$(depends): %.d: %.cpp
	set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

include $(depends)

gple.x: $(objects)
	$(CXX) $^ $(LDFLAGS) -o $@ $(LDLIBS)
$(objects): %.o: %.cpp $(PCH)
	$(CXX) -c $< $(CXXFLAGS) -o $@
$(PCH): stdafx.h
	$(CXX) stdafx.h $(CXXFLAGS)

.PHONY: clean
clean:
	\rm -f *.o *.d* *.gch

.PHONY: distclean
distclean: clean
	\rm -rf -- *log *.out* *.png *.gif gp_test core.* *.txt Doxyfile* html/ latex/

.PHONY: format
format:
	for f in *.c* *.h* ; \
	do \
		clang-format --style=file -i $${f} ; \
	done
