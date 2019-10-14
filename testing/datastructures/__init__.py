
#Make it look like a package
from glob import glob
from os import environ
from os.path import basename, dirname
from pdb import set_trace

#gather all the files here
modules = [basename(i.replace('.py','')) for i in glob('%s/[A-Za-z]*.py' % dirname(__file__))]
__all__ = []
structure_list=[]
for module_name in modules:
    module = __import__(module_name, globals(), locals(), [module_name])
    for model_name in [i for i in dir(module) if 'TrainData' in i]:
        
        
        model = getattr(module, model_name)
        globals()[model_name] = model
        locals( )[model_name] = model
        __all__.append(model_name)
        structure_list.append(model_name)

