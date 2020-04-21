import os

def DJCSetGPUs(gpus_string: str =""):
    if not len(gpus_string):
        import imp
        try:
            imp.find_module('setGPU')
            import setGPU
        except Exception as e:
            print('DeepJetCore.DJCSetGPU: no GPU specified and automatic setting impossible')
            raise e
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_string
        print('running on GPU(s) '+gpus_string)