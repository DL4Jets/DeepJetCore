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

INTERFACE=./interface

# Helper files for modules
SWIG_WRAPPERS := $(wildcard cxx/*.cxx)
SWIG_OBJ_FILES := $(addprefix ./swigobj/, $(notdir $(SWIG_WRAPPERS:.cxx=.o)))

HELPERS := $(wildcard cpp/*.cpp)
HELPERS_OBJ_FILES := $(addprefix ./obj/, $(notdir $(HELPERS:.cpp=.o)))
HELPERS_SHARED_LIBS := $(addprefix ./, $(notdir $(HELPERS:.cpp=.so)))

# DeepJet modules
MODULES := $(wildcard src/*.C)
MODULES_OBJ_FILES := $(addprefix ./, $(notdir $(MODULES:.C=.o)))
MODULES_SHARED_LIBS := $(addprefix ./, $(notdir $(MODULES:.C=.so)))

UNAME_S := $(shell uname -s)
# Remove Linux flags in OSX
ifeq ($(UNAME_S),Darwin)
    LINUXADD=""
endif

all: libquicklz.so $(HELPERS_OBJ_FILES) $(SWIG_OBJ_FILES) $(HELPERS_SHARED_LIBS) $(MODULES_OBJ_FILES) $(MODULES_SHARED_LIBS)
# all: libquicklz.so obj/%.o lib/%.so %.o %.so

# quicklz for compression
libquicklz.so: src/quicklz.c
	gcc -Wall -O2 -fPIC -shared -I$(INTERFACE) -I$(PYTHON_INCLUDE) -L$(PYTHON_LIB) -lpython2.7 $< -o $@

swigobj/%.o: cxx/%.cxx libquicklz.so
	g++ -Wall -fPIC $(CFLAGS) $(ROOTSTUFF) -I$(PYTHON_INCLUDE) -I$(INTERFACE) -I$(BOOST_INC) -c -o $@ $< 

obj/%.o: cpp/%.cpp libquicklz.so
	g++ -Wall -fPIC $(CFLAGS) $(ROOTSTUFF) -I$(INTERFACE) -c -o $@ $< 

# compile helpers with wrappers in shared lib
indata.so: obj/indata.o swigobj/indata_wrap.o
	g++ -Wall -shared $(LINUXADD) $(ROOTSTUFF) -L./ -lquicklz -L$(PYTHON_LIB) -lpython2.7 $^ -o $@

helper.so: obj/helper.o swigobj/helper_wrap.o
	g++ -Wall -shared $(LINUXADD) $(ROOTSTUFF) -L./ -lquicklz -L$(PYTHON_LIB) -lpython2.7 $^ -o $@

friendTreeInjector.so: obj/friendTreeInjector.o swigobj/friendTreeInjector_wrap.o
	g++ -Wall -shared $(LINUXADD) $(ROOTSTUFF) -L./ -lquicklz -L$(PYTHON_LIB) -lpython2.7 $^ -o $@

rocCurve.so: obj/rocCurve.o swigobj/rocCurve_wrap.o
	g++ -Wall -shared $(LINUXADD) $(ROOTSTUFF) -L./ -lquicklz -L$(PYTHON_LIB) -lpython2.7 $^ -o $@

rocCurveCollection.so: obj/rocCurveCollection.o swigobj/rocCurveCollection_wrap.o
	g++ -Wall -shared $(LINUXADD) $(ROOTSTUFF) -L./ -lquicklz -L$(PYTHON_LIB) -lpython2.7 $^ -o $@

%.o: src/%.C $(HELPERS_SHARED_LIBS) 
	g++ -Wall -fPIC $(ROOTSTUFF) -O2 -I$(INTERFACE) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -c $< -o $@ 
%.so: %.o  $(HELPERS_SHARED_LIBS)
	g++ -Wall -shared $(LINUXADD) $(ROOTSTUFF) -L$(BOOST_LIB) -lboost_python -L$(PYTHON_LIB) -lpython2.7 -L./ -lquicklz -l:indata.so -l:rocCurve.so -l:rocCurveCollection.so -l:helper.so -l:friendTreeInjector.so $< -o $@ 

# Avoiding future filename clashes
.PHONY: clean 

clean: 
	rm -f libquicklz.so *.o *.so $(SWIG_OBJ_FILES) $(HELPERS_OBJ_FILES) $(HELPERS_SHARED_LIBS) $(MODULES_OBJ_FILES) $(MODULES_SHARED_LIBS) 
