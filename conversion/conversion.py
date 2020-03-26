try:
    import datastructures
    from datastructures import *
except ImportError:
    print('datastructure modules not found. Please define a DeepJetCore submodule')

class_options=[]
import inspect, sys
for name, obj in inspect.getmembers(sys.modules['datastructures']):
    if inspect.isclass(obj) and 'TrainData' in name:
        class_options.append(obj)
      
class_options = dict((str(i).split("'")[1].split('.')[-1], i) for i in class_options)
