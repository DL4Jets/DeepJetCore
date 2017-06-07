


from testing import makeROCs_async, testDescriptor
#from keras.models import load_model
from DataCollection import DataCollection
from argparse import ArgumentParser

parser = ArgumentParser('make a set of ROC curves, comparing two training')
parser.add_argument('inputTextFileA')
parser.add_argument('inputTextFileB')
parser.add_argument('outputDir')
args = parser.parse_args()


args.outputDir+='/'


btruth='isB+isBB+isLeptonicB+isLeptonicB_C'



textfiles=[args.inputTextFileA,args.inputTextFileB,args.inputTextFileA,args.inputTextFileB]



print('creating ROCs')


makeROCs_async(textfiles,['file 0 B vs. l', #legend names
                          'file 1 B vs. l',
                          'file 0 B vs. C',
                          'file 1 B vs. C'],
         ['0.prob_isB+0.prob_isBB', #probabiliies with indecies for each input file
          '1.prob_isB+1.prob_isBB',
          '0.prob_isB+0.prob_isBB',
          '1.prob_isB+1.prob_isBB'],
          [btruth,btruth,btruth,btruth], #signal truth
         ['isUD+isS+isG', #compare to
          'isUD+isS+isG',
          'isC',
          'isC'],
         ['darkblue', #line colors and styles
          'darkgreen',
          'blue,dashed',
          'green,dashed'],
         args.outputDir+"comp_pt30-100.pdf",
         'jet_pt>30&&jet_pt<100') #cuts




