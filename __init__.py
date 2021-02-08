

        
import sys
import tensorflow as tf
sys.modules["keras"] = tf.keras

__version__ = '3.2'

#shortcuts 
from .TrainData import TrainData
from .SimpleArray import SimpleArray
from .DataCollection import DataCollection
from .Weighter import Weighter

