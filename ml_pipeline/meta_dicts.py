from torch.optim import Adam, AdamW, RAdam, SGD
from torch.nn import CrossEntropyLoss

from models.CNN_Attention import CNN_Attention
from models.CNN_BILSTM import CNN_BILSTM
from models.CNN_Transformers import CNN_T
from models.M5 import M5Net
from models.M5_enhanced import M5ENet
from models.ResNet1D import ResNet1D
from models.soundnet import SoundNet
from feature_extractors.main_fe import MelSpec_FE, MFCC_FE, Raw_FE


# Available feature types
FEATURES = {"mel":  {"type": "2d", "cls": MelSpec_FE}, 
            "mfcc": {"type": "2d", "cls": MFCC_FE}, 
            "raw":  {"type": "1d", "cls": Raw_FE}}

# Model_id : (it's class, feature type)
MODELS = {
    "CB":   {"cls": CNN_BILSTM, "feature": "2d"},
    "CA":   {"cls": CNN_Attention, "feature": "2d"},
    "M5":   {"cls": M5Net, "feature": "1d"},
    "M5E":  {"cls": M5ENet, "feature": "1d"},
    "R1D":  {"cls": ResNet1D, "feature": "1d"},
    "SN":   {"cls": SoundNet, "feature": "1d"},
    "CNNT": {"cls": CNN_T, "feature": "2d"}
}

LOSS_FUNCS = {
    "crossentropy": CrossEntropyLoss,
}

OPTIMS = {
    "adam": Adam,
    "adamw": AdamW,
    "radam": RAdam,
    "sgd": SGD
}