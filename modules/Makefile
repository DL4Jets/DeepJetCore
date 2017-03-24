# location of the Python header files
 
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = ${CONDA_PREFIX}/include/python2.7
 
# location of the Boost Python include files and library
 
# also works on gpu in compiled version
# this is just luck, ...
BOOST_INC = ${CONDA_PREFIX}/include
BOOST_LIB = ${CONDA_PREFIX}/lib
LINUXADD=-Wl,--export-dynamic
ROOTSTUFF=`root-config --cflags --libs --glibs` -g

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

MODULES := $(wildcard src/*.C)
MODULES_OBJ_FILES := $(addprefix ./,$(notdir $(MODULES:.C=.o)))
MODULES_SHARED_LIBS := $(addprefix ./,$(notdir $(MODULES:.C=.so)))


all: $(MODULES_SHARED_LIBS)

#helpers

obj/%.o: src/%.cpp
	g++ $(ROOTSTUFF) -I./interface -O2 -fPIC -c -o $@ $< 
    
#pack helpers in lib
libdeepjethelpers.so: $(OBJ_FILES)
	g++ -shared $(LINUXADD)  $(ROOTSTUFF) obj/*.o -o $@ 
  

%.so: %.o libdeepjethelpers.so
	g++ -shared $(LINUXADD)  $(ROOTSTUFF) -L./ -ldeepjethelpers -L$(BOOST_LIB)  -lboost_python -L${CONDA_PREFIX}/lib/python$(PYTHON_VERSION)/config -lpython2.7  $< -o $(@) 


%.o: src/%.C 
	g++   $(ROOTSTUFF) -O2 -I./interface -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -o $(@) $<


 
   
clean: 
	rm -f $(OBJ_FILES) $(SHARED_LIBS) $(MODULES_SHARED_LIBS) $(MODULES_OBJ_FILES) libdeepjethelpers.so