'''
Created on 4 Mar 2017

@author: jkiesele
'''

import time as tm
class stopwatch(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.start= tm.time()
        
        
    def getAndReset(self):
        nowT=tm.time()
        ret = (nowT - self.start)
        self.start= tm.time()
        return ret
        
    def getAndContinue(self):
        nowT=tm.time()
        return (nowT - self.start)
        