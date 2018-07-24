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

QUICKLZ_SRC := ./src/quicklzpy.c

# Helper files for modules
HELPERS := $(wildcard src/*.cxx)
HELPERS_OBJ_FILES := $(addprefix obj/,$(notdir $(HELPERS:.cxx=.o)))
HELPERS_SHARED_LIBS := $(addprefix ./obj,$(notdir $(HELPERS_OBJ_FILES:.o=.so)))

# DeepJet modules
MODULES := $(wildcard src/*.C)
MODULES_OBJ_FILES := $(addprefix ./,$(notdir $(MODULES:.C=.o)))
MODULES_SHARED_LIBS := $(addprefix ./,$(notdir $(MODULES:.C=.so)))

UNAME_S := $(shell uname -s)
# Remove Linux flags in OSX
ifeq ($(UNAME_S),Darwin)
    LINUXADD=""
endif

all: libquicklz.so $(HELPERS_OBJ_FILES) $(HELPERS_SHARED_LIBS) $(MODULES_OBJ_FILES) $(MODULES_SHARED_LIBS)

# quicklz for compression
libquicklz.so: $(QUICKLZ_SRC)
	gcc -shared -I./interface -O2 -fPIC $< -o $@

$(HELPERS_OBJ_FILES): $(HELPERS)
	g++ $(CFLAGS) $(ROOTSTUFF) -I./interface -O2 -fPIC -c -o $@ $< 

# pack helpers in shared lib
$(HELPERS_SHARED_LIBS): $(HELPERS_OBJ_FILES)
	g++ -shared $(LINUXADD) $(ROOTSTUFF) $< -o $@

$(MODULES_OBJ_FILES): $(MODULES) 
	g++ $(ROOTSTUFF) -O2 -I./interface -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c $< -o $@ 

$(MODULES_SHARED_LIBS): $(MODULES_OBJ_FILES) $(HELPERS_SHARED_LIBS) libquicklz.so
	g++ -shared $(LINUXADD) $(ROOTSTUFF) -lquicklz -L./ -L$(BOOST_LIB)  -lboost_python -L$(PYTHON_LIB) -lpython2.7  $< -o $@ 

# Avoiding future filename clashes
.PHONY: clean 

clean: 
	rm -f libquicklz.so $(HELPERS_OBJ_FILES) $(HELPERS_SHARED_LIBS) $(MODULES_OBJ_FILES) $(MODULES_SHARED_LIBS) 