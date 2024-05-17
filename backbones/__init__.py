from backbones.PSMNet.stackhourglass import PSMNet
from backbones.GwcNet.gwcnet import GwcNet_G, GwcNet_GC
from backbones.GANet.GANet_deep import GANet



__models__ = {
    "PSMNet": PSMNet,
    "GwcNet_G": GwcNet_G,
    "GwcNet_GC": GwcNet_GC,
    "GANet":GANet, 
    }
