import sys

if  'wandb' in sys.modules:
    print("you have imported wandb before importing the DeepJetCore.wandb_interface. Some functionality may be broken.")

import wandb

'''
Global wandb object to be accessed from different modules.

Please always use wherever possible:

from DeepJetCore.wandb_interface import wandb_wrapper as wandb

'''

class _WandbWrapper:
    def __init__(self):
        self._wandb = None
        self.active = True #for future use
        self.log_cache = {}
        self.debug_mode = False

    def init(self, *args, **kwargs):
        if not self.active:
            return
        if self._wandb is not None:
            raise RuntimeError("wandb is already initialised")
        self._wandb = wandb 
        self._wandb.init(*args, **kwargs)

    def wandb(self):
        return self._wandb

    def _dummy(self, *args, **kwargs):
        pass

    def log(self, my_dict: dict):
        if not self.active:
            return
        #check if keys are already in the cache
        
        if self.debug_mode:
            for key in my_dict.keys():
                if key in self.log_cache:
                    raise ValueError(f"Key '{key}' already in cache. Please flush the cache before logging again.")
            
        self.log_cache.update(my_dict)

    def flush(self):
        if not self.active:
            return
        if self._wandb is not None:
            self._wandb.log(self.log_cache)
            self.log_cache.clear()

    def __getattr__(self, name):
        if not self.active:
            return self._dummy
        if self._wandb is not None:
            return getattr(self._wandb, name)
                
        raise AttributeError(f"WandB instance not initialized and set to active at the same time. Attribute '{name}' is not available.")


wandb_wrapper = _WandbWrapper()


# run a quick test
if __name__ == '__main__':
    

    #wandb_wrapper.init(project='test')

    for i in range(10):
        wandb_wrapper.log({'test': i})
        wandb_wrapper.log({'testB': 2+i})
        wandb_wrapper.log({'testC': 3+i})
        wandb_wrapper.flush()

    
    print("done")