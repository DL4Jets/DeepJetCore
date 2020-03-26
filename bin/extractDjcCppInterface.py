#!/usr/bin/env python3


from argparse import ArgumentParser
import os

parser = ArgumentParser('Extract the C++ interface for trainData etc to be used outside in a simple package')
parser.add_argument('outputDir')


args = parser.parse_args()

 
script = '''
#!/bin/bash
mkdir -p {outdir}
mkdir -p {outdir}/interface
mkdir -p {outdir}/src
mkdir -p {outdir}/obj
cp $DEEPJETCORE/compiled/interface/version.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/IO.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/quicklz.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/quicklzWrapper.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/simpleArray.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/trainData.h {outdir}/interface/
cp $DEEPJETCORE/compiled/src/quicklz.c {outdir}/src/

'''.format(outdir=args.outputDir)

os.system(script)

makefile = '''

ROOTLIBS=`root-config --libs --glibs --ldflags`
ROOTCFLAGS=`root-config  --cflags`
CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))

BINS := $(patsubst bin/%.cpp, %, $(wildcard bin/*.cpp))


all: $(patsubst bin/%.cpp, %, $(wildcard bin/*.cpp)) libquicklz.so libdeepjetcoredataformats.so

#helpers
libquicklz.so:
	gcc -shared -O2 -fPIC src/quicklz.c -o libquicklz.so
    
obj/%.o: src/%.cpp
	g++ $(CFLAGS) $(ROOTCFLAGS) -I./interface -O2 -fPIC -c -o $@ $< 

#pack helpers in lib
libdeepjetcoredataformats.so: $(OBJ_FILES) 
	g++ -o $@ -shared -fPIC  -fPIC  $(OBJ_FILES) $(ROOTLIBS)


%: bin/%.cpp libdeepjetcoredataformats.so libquicklz.cxx
	g++ $(CFLAGS) -I./interface  $< -L. -ldeepjetcoredataformats -lquicklz  $(ROOTCFLAGS) $(ROOTLIBS)   -o  $@  
    

clean: 
	rm -f libdeepjetcoredataformats.so libquicklz.so
	rm -f obj/*.o 

'''


with  open(args.outputDir+'/Makefile','w') as lfile:
    lfile.write(makefile)

