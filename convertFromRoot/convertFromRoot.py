#!/usr/bin/env python
# encoding: utf-8
'''
convertFromRoot -- converts the root files produced with the deepJet ntupler to the data format used by keras for the DNN training

convertFromRoot is a small program that converts the root files produced with the deepJet ntupler to the data format used by keras for the DNN training


@author:     jkiesele

'''

import sys
import os

from argparse import ArgumentParser
from pdb import set_trace
import logging
logging.getLogger().setLevel(logging.INFO)

__all__ = []
__version__ = 0.1
__date__ = '2017-02-22'
__updated__ = '2017-02-22'

DEBUG = 0
TESTRUN = 0
PROFILE = 0

def main(argv=None):
    '''Command line options.'''

    program_name = os.path.basename(sys.argv[0])
    program_version = "v0.1"
    program_build_date = "%s" % __updated__

    program_version_string = '%%prog %s (%s)' % (program_version, program_build_date)
    program_longdesc = ''' ''' # optional - give further explanation about what the program does
    program_license = "Copyright 2017 user_name (organization_name) Licensed under the Apache License 2.0\nhttp://www.apache.org/licenses/LICENSE-2.0"

    
    #try: 
        # setup option parser
    from TrainData import TrainData
    from TrainData_deepCSV import TrainData_deepCSV, TrainData_deepCSV_RNN, TrainData_deepCSV_RNN_Deeper
    from TrainData_deepConvCSV import TrainData_deepConvCSV
    from TrainData_deepCMVA import TrainData_deepCMVA
    from TrainData_deepCSV_PF import TrainData_deepCSV_PF,TrainData_deepCSV_miniPF,TrainData_deepCSV_microPF,TrainData_deepCSV_softL_PF,  TrainData_deepCSV_PF_rec
    from TrainData_deepCSV_PF_Reg import TrainData_deepCSV_PF_Reg
    from TrainData_deepJet_Reg import TrainData_deepJet_Reg, TrainData_PF_Reg
    from TrainData_deepCSV_PF_binned import TrainData_deepCSV_PF_Binned
    from TrainData_deepFlavour import TrainData_deepFlavour_FT_reg_noScale,TrainData_deepFlavour_QGOnly_reg,TrainData_deepFlavour_FT,TrainData_deepFlavour_FT_reg,TrainData_deepFlavour_FT_map,TrainData_deepFlavour_FT_map_reg,TrainData_image, TrainData_deepFlavour_cleaninput, TrainData_deepFlavour_cleanBTVOnly, TrainData_deepFlavour_nopuppi
    from TrainData_FatJet import TrainData_FatJet_Test
    from TrainData_PT_recur import TrainData_PT_recur, TrainData_QG_simple, TrainData_recurrent_fullTruth
    from TrainData_deepCSV_int import TrainData_deepCSV_int,TrainData_deepCSV_conv
    from TrainData_deepAK8 import TrainData_AK8Jet_init 
    
    class_options = [
        TrainData_deepCSV_RNN,
        TrainData_deepCSV_RNN_Deeper,
        TrainData_deepCSV,
        TrainData_deepConvCSV,
        TrainData_deepCMVA,
        TrainData_deepCSV_PF,
        TrainData_deepCSV_miniPF,
        TrainData_deepCSV_microPF,
        TrainData_deepCSV_softL_PF,
        TrainData_deepCSV_PF_Reg,
        TrainData_deepJet_Reg, 
        TrainData_PF_Reg,
        TrainData_deepCSV_PF_Binned,
        TrainData_deepFlavour_FT,
        TrainData_deepFlavour_FT_reg,
        TrainData_deepFlavour_FT_reg_noScale,
        TrainData_deepFlavour_FT_map,
        TrainData_deepFlavour_QGOnly_reg,
        TrainData_image,
        TrainData_deepCSV_PF_rec,
        TrainData_PT_recur,
        TrainData_FatJet_Test,
        TrainData_QG_simple,
        TrainData_recurrent_fullTruth,
        TrainData_deepFlavour_FT_map_reg,
        TrainData_deepCSV_int,
        TrainData_deepCSV_conv,
        TrainData_AK8Jet_init,
        TrainData_deepFlavour_cleaninput,
        TrainData_deepFlavour_cleanBTVOnly,
			  TrainData_deepFlavour_nopuppi,
        ]
    class_options = dict((str(i).split("'")[1].split('.')[-1], i) for i in class_options)

    parser = ArgumentParser('program to convert root tuples to traindata format')
    parser.add_argument("-i", help="set input sample description (output from the check.py script)", metavar="FILE")
    parser.add_argument("-o",  help="set output path", metavar="PATH")
    parser.add_argument("-c",  choices = class_options.keys(), help="set output class (options: %s)" % ', '.join(class_options.keys()), metavar="Class")
    parser.add_argument("-r",  help="set path to snapshot that got interrupted", metavar="FILE", default='')
    parser.add_argument("-n", default='', help="(optional) number of child processes")
    parser.add_argument("--testdatafor", default='')
    parser.add_argument("--usemeansfrom", default='')
    parser.add_argument("--nothreads", action='store_true')
    parser.add_argument("--means", action='store_true', help='compute only means')
    parser.add_argument("--batch", help='Provide a batch ID to be used')
    parser.add_argument("-v", action='store_true', help='verbose')
    parser.add_argument("-q", action='store_true', help='quiet')
    
    # process options
    args=parser.parse_args()
    infile=args.i
    outPath=args.o
    class_name=args.c    
    recover=args.r
    testdatafor=args.testdatafor
    usemeansfrom=args.usemeansfrom
    nchilds=args.n

    if args.batch and not (args.usemeansfrom or args.testdatafor):
        raise ValueError(
            'When running in batch mode you should also '
            'provide a means source through the --usemeansfrom option'
            )

    if args.v:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.q:
        logging.getLogger().setLevel(logging.WARNING)

    if infile:
        logging.info("infile = %s" % infile)
    if outPath:
        logging.info("outPath = %s" % outPath)

    # MAIN BODY #
    from DataCollection import DataCollection
    dc = DataCollection(nprocs = (1 if args.nothreads else -1))  
    if len(nchilds):
        dc.nprocs=int(nchilds)  
    
    if class_name in class_options:
        traind = class_options[class_name]
    elif not recover and not testdatafor:
        raise Exception('wrong class selecton') #should never really happen as we catch it in the parser        
    if testdatafor:
        logging.info('converting test data, no weights applied')
        dc.createTestDataForDataCollection(
            testdatafor, infile, outPath, 
            outname = args.batch if args.batch else 'dataCollection.dc',
            batch_mode = bool(args.batch)
        )    
    elif recover:
        dc.recoverCreateDataFromRootFromSnapshot(recover)        
    elif args.means:
        dc.convertListOfRootFiles(
            infile, traind(), outPath, 
            means_only=True, output_name='batch_template.dc'
            )
    else:
        dc.convertListOfRootFiles(
            infile, traind(), outPath, 
            usemeansfrom, output_name = args.batch if args.batch else 'dataCollection.dc',
            batch_mode = bool(args.batch)
            )
    


#if __name__ == "__main__":
if DEBUG:
    sys.argv.append("-h")
if TESTRUN:
    import doctest
    doctest.testmod()
if PROFILE:
    import cProfile
    import pstats
    profile_filename = 'convertFromRoot_profile.txt'
    cProfile.run('main()', profile_filename)
    statsfile = open("profile_stats.txt", "wb")
    p = pstats.Stats(profile_filename, stream=statsfile)
    stats = p.strip_dirs().sort_stats('cumulative')
    stats.print_stats()
    statsfile.close()
    sys.exit(0)
sys.exit(main())
