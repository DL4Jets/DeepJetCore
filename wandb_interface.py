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
        '''
        Gives access to the wandb object.
        '''
        return self._wandb

    def _dummy(self, *args, **kwargs):
        pass

    def log(self, my_dict: dict):
        '''
        Use in the same way as wandb.log. 
        The difference is that the values are not immediately logged to wandb, but stored in a cache.
        '''
        if not self.active:
            return
        #check if keys are already in the cache
        
        if self.debug_mode:
            for key in my_dict.keys():
                if key in self.log_cache:
                    raise ValueError(f"Key '{key}' already in cache. Please flush the cache before logging again.")
            
        self.log_cache.update(my_dict)

    def flush(self, prefix : str =''):
        '''
        logs all collected information to wandb.
        Parameters:
        prefix: str, optional
            prefix to be prepended to all keys in the cache. A '_' will be added between the prefix and the key.
        '''
        if not self.active:
            return
        if self._wandb is not None:
            if len(prefix):
                log_cache = {f'{prefix}_{key}': value for key, value in self.log_cache.items()}
                self._wandb.log(log_cache)
            else:
                self._wandb.log(self.log_cache)
            self.log_cache.clear()

    #just a convenience function
    def flush_val(self):
        self.flush('val')


wandb_wrapper = _WandbWrapper()


# a little test to log some stuff and flush it to wandb
def test():
    wandb_wrapper.init(project='test')
    counter = 0
    for _ in range(5):
        for i in range(50):
            wandb_wrapper.log({'test': counter})
            wandb_wrapper.flush()
            counter += 1
        temp_counter = counter
        #now for val
        for i in range(10):
            wandb_wrapper.log({'test': counter})
            wandb_wrapper.flush_val()
            #add some wiggles to the counter using % on i
            add = i%2 - 0.5
            counter += add
        counter = temp_counter

