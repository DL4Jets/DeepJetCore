# location of the Python header files
 
PYTHON_VERSION = 3.4
PYTHON_INCLUDE = ${CONDA_PREFIX}/include/python3.4m
 
# location of the Boost Python include files and library
 
# also works on gpu in compiled version
# this is just luck, ...
BOOST_INC = ${CONDA_PREFIX}/include
BOOST_LIB = ${CONDA_PREFIX}/lib
LINUXADD=-Wl,--export-dynamic
ROOTSTUFF=`root-config --cflags --libs` -g

CPP_FILES := $(wildcard src/*.C)
HELPER_CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix ./,$(notdir $(CPP_FILES:.C=.o)))
SHARED_LIBS := $(addprefix ./,$(notdir $(CPP_FILES:.C=.so)))

# compile mesh classes
# all: $(patsubst %.C, %, $(wildcard *.C))
# all: $(patsubst ./%.C, %, $(wildcard ./*.C))

all: $(SHARED_LIBS)


%.so: %.o libdeepjethelpers.so
	g++ -shared $(LINUXADD)  $(ROOTSTUFF) -L./ -ldeepjethelpers -L$(BOOST_LIB)  -lboost_python3 -L${CONDA_PREFIX}/lib/python$(PYTHON_VERSION)/config-3.4m -lpython3.4m  $< -o $(@) 


%.o: src/%.C
	g++ -D__USE_XOPEN2K8  $(ROOTSTUFF) -O2 -I./interface -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -o $(@) $<
    
libdeepjethelpers.so: helpers.o
	g++ -shared $(LINUXADD)  $(ROOTSTUFF)  -L$(BOOST_LIB)  -lboost_python3 -L${CONDA_PREFIX}/lib/python$(PYTHON_VERSION)/config-3.4m -lpython3.4m  $< -o $(@) 

helpers.o: $(HELPER_CPP_FILES)
	g++ -D__USE_XOPEN2K8  $(ROOTSTUFF) -O2 -I./interface -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -o $(@) $<
    
 
   
clean: 
	rm -f $(OBJ_FILES) $(SHARED_LIBS) helpers.o libdeepjethelpers.so