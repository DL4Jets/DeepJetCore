# location of the Python header files

PYTHON_VERSION = 2.7
# location of the Boost Python include files and library
PYTHON_INCLUDE = ${CONDA_PREFIX}/include/python2.7
PYTHON_LIB=${CONDA_PREFIX}/lib/python$(PYTHON_VERSION)/config
# also works on gpu in compiled version
# this is just luck, ...
BOOST_INC = ${CONDA_PREFIX}/include
BOOST_LIB = ${CONDA_PREFIX}/lib
ifdef CMSSW_BASE
	PYTHON_BASE=$(shell scram tool info python | grep BASE | cut -d "=" -f 2)
	PYTHON_INCLUDE=${PYTHON_BASE}/include/python${PYTHON_VERSION}/
	PYTHON_LIB=${PYTHON_BASE}/lib/python${PYTHON_VERSION}/config
	BOOST_BASE=$(shell scram tool info boost | grep BASE | cut -d "=" -f 2)
	BOOST_INC = ${BOOST_BASE}/include
	BOOST_LIB = ${BOOST_BASE}/lib
endif

LINUXADD=-Wl,--export-dynamic
ROOTSTUFF=`root-config --cflags --libs --glibs` -g
CFLAGS=

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

BINS := $(patsubst bin/%.cpp, %, $(wildcard bin/*.cpp))

MODULES := $(wildcard src/*.C)
MODULES_OBJ_FILES := $(addprefix ./,$(notdir $(MODULES:.C=.o)))
MODULES_SHARED_LIBS := $(addprefix ./,$(notdir $(MODULES:.C=.so)))

UNAME_S := $(shell uname -s)
# remove linux flags in osx
ifeq ($(UNAME_S),Darwin)
	LINUXADD=""
endif


all: $(MODULES_SHARED_LIBS) $(patsubst to_bin/%.cpp, %, $(wildcard to_bin/*.cpp)) classdict.so

classdict.cxx: src/LinkDef.h
	rootcling -v4 -f $@  -rmf classdict.rootmap -rml classdict.so  $^

classdict.so: classdict.cxx
	g++ -std=c++11 -fPIC -shared -o classdict.so classdict.cxx 	
	
#make_example_data: to_bin/make_example_data.cpp libdeepjetcorehelpers.so classdict.cxx
#	g++ $(CFLAGS) $(LINUXADD) $(ROOTSTUFF) -I./interface  $< -o $@
#	mv make_example_data ../bin/
	
	
%: to_bin/%.cpp libdeepjetcorehelpers.so classdict.cxx
	g++ $(CFLAGS) $(LINUXADD) $(ROOTSTUFF) -I./interface  $< -o $@ 
	mv $@ ../bin/
	
#helpers
libquicklz.so:
	gcc -shared -O2 -fPIC src/quicklz.c -o libquicklz.so

obj/%.o: src/%.cpp
	g++ $(CFLAGS) $(ROOTSTUFF) -I./interface -O2 -fPIC -c -o $@ $< 

#pack helpers in lib
libdeepjetcorehelpers.so: $(OBJ_FILES) 
	g++ -shared $(LINUXADD) -fPIC $(ROOTSTUFF) obj/*.o -o $@ 

%.so: %.o libdeepjetcorehelpers.so libquicklz.so
	g++ -shared $(LINUXADD)  $(ROOTSTUFF) -lquicklz -L./ -ldeepjetcorehelpers -L$(BOOST_LIB)  -lboost_python -L$(PYTHON_LIB) -lpython2.7  $< -o $(@) 


%.o: src/%.C 
	g++   $(ROOTSTUFF) -O2 -I./interface -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -o $(@) $<



clean: 
	rm -f $(OBJ_FILES) $(SHARED_LIBS) $(MODULES_SHARED_LIBS) $(MODULES_OBJ_FILES) libdeepjetcorehelpers.so libquicklz.so classdict.so classdict.cxx classdict.rootmap classdict_rdict.pcm