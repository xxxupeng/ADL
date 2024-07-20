from backbones.PSMNet.stackhourglass import PSMNet
from backbones.GwcNet.gwcnet import GwcNet_G, GwcNet_GC

__models__ = {
    "PSMNet": PSMNet,
    "GwcNet_G": GwcNet_G,
    "GwcNet_GC": GwcNet_GC,
    }

try:
    from backbones.GANet.GANet_deep import GANet
    __models__["GANet"] = GANet
except:
    print('If you want to train the GANet, please install the [NVIDIA-Apex package](https://github.com/NVIDIA/apex) and compile the [GANet libs](https://github.com/feihuzhang/GANet).')