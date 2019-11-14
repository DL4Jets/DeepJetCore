#!/usr/bin/env python


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
cp $DEEPJETCORE/compiled/interface/version.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/IO.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/quicklz.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/quicklzWrapper.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/simpleArray.h {outdir}/interface/
cp $DEEPJETCORE/compiled/interface/trainData.h {outdir}/interface/
cp $DEEPJETCORE/compiled/src/quicklz.c {outdir}/src/

'''.format(outdir=args.outputDir)

os.system(script)