.RECIPEPREFIX = ~
GENERATES = main
TRASH = layers

CXX =  mpicxx.mpich
CXXFLAGS = -O3

INCLUDE  = -Isrc/
SRC = \
   $(wildcard src/*.cpp)

all: main

main: ${SRC}
~     ${CXX} $^ ${CXXFLAGS} -o $@
~     mkdir -p layers

clean:
~      rm -rf $(TRASH)

distclean: clean
~          rm -rf $(GENERATES)
