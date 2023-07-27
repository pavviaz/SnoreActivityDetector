import json
import os
import torch
from models.CNN_Attention import CNN_Attention
from models.M5 import M5Net
from models.CNN_BILSTM import CNN_BILSTM
from models.soundnet import SoundNet

# Meta file name
META_FILE_NAME = "meta.json"

# Available feature types
FEATURES_2D = ["mel", "mfcc"]
FEATURES_1D = ["raw"]

# Model_id : (it's class, if model uses 2D features or not)
MODELS = {"CB": (CNN_BILSTM, True),
          "CA": (CNN_Attention, True),
          "M5": (M5Net, False),
          "SN": (SoundNet, False),}


class ModelLoader:
    def __init__(self, path=None, feature_type=None, model_type=None, sample_rate=None, device="cpu", **kwargs):
        try:
            if path:
                with open(os.path.join(path, META_FILE_NAME), "r") as meta:
                    meta = json.load(meta)
                self.feature_type = meta["features_type"]
                self.model_type = meta["model_type"]
                self.sample_rate = meta["sample_rate"]
            else:
                self.feature_type = feature_type
                self.model_type = model_type
                self.sample_rate = sample_rate
            
            if self.model_type in MODELS.keys():
                if (MODELS[self.model_type][1] and self.feature_type in FEATURES_2D) \
                or (not MODELS[self.model_type][1] and self.feature_type in FEATURES_1D):
                    self.model = MODELS[self.model_type][0](**kwargs).to(device)
                else:
                    raise ValueError("Passing not appropriate feature type for the model")
            else:
                raise ValueError("No such model")
            
            if path:
                model_path = os.path.join(path, f"{os.path.basename(os.path.normpath(path))}.pt")
                self.model.load_state_dict(torch.load(model_path), strict=False)
        except:
            raise IOError
        
        self.model.eval()        
        
    def predict(self, input):
        return self.model(input) if not self.feature_type == "raw" else self.model(input).squeeze(1)