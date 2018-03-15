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
CFLAGS=
CPP_FILES := $(wildcard DeepJetCore/compiled/src/*.cpp)
OBJ_FILES := $(addprefix ./,$(notdir $(CPP_FILES:.cpp=.o)))

MODULES := $(wildcard DeepJetCore/compiled/src/*.C)
MODULES_OBJ_FILES := $(addprefix ./,$(notdir $(MODULES:.C=.o)))
MODULES_SHARED_LIBS := $(addprefix ./,$(notdir $(MODULES:.C=.so)))

UNAME_S := $(shell uname -s)
# remove linux flags in osx
ifeq ($(UNAME_S),Darwin)
	LINUXADD=""
endif

all: $(MODULES_SHARED_LIBS)

#helpers
DeepJetCore/compiled/libquicklz.so:
	gcc -shared -O2 -fPIC DeepJetCore/compiled/src/quicklz.c -o ./DeepJetCore/compiled/libquicklz.so

DeepJetCore/compiled/obj/%.o: DeepJetCore/compiled/src/%.cpp
	g++ $(CFLAGS) $(ROOTSTUFF) -I./DeepJetCore/compiled/interface -O2 -fPIC -c -o $@ $< 
    
#pack helpers in lib
DeepJetCore/compiled/libdeepjetcorehelpers.so: $(OBJ_FILES)
	g++ -shared $(LINUXADD)  $(ROOTSTUFF) DeepJetCore/compiled/obj/*.o -o $@ 
  

DeepJetCore/compiled/%.so: DeepJetCore/compiled/%.o DeepJetCore/compiled/libdeepjetcorehelpers.so DeepJetCore/compiled/libquicklz.so
	g++ -shared $(LINUXADD)  $(ROOTSTUFF) -lquicklz -L./ -ldeepjetcorehelpers -L$(BOOST_LIB)  -lboost_python -L${CONDA_PREFIX}/lib/python$(PYTHON_VERSION)/config -lpython2.7  $< -o $(@) 


DeepJetCore/compiled/%.o: DeepJetCore/compiled/rc/%.C 
	g++   $(ROOTSTUFF) -O2 -I./interface -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -c -o $(@) $<


clean: 
	rm -f $(OBJ_FILES) $(SHARED_LIBS) $(MODULES_SHARED_LIBS) $(MODULES_OBJ_FILES) DeepJetCore/compiled/libdeepjetcorehelpers.so DeepJetCore/compiled/libquicklz.so
	
	
	
