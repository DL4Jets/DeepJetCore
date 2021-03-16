from DeepJetCore.compiled.c_testFunctions import *

import unittest


class TestCFunctions(unittest.TestCase):
    def test_trainDataFiller(self):
        testTrainDataFileStreamer()
    