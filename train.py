import json
import os
import shutil
import time
from munch import munchify
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.checkpointing import Checkpointing
from utils.logger_init import log_init
from model_loader import ModelLoader


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRAINED_MODELS_DIR = "trained_models"
CKPT_DIR = "checkpoints"
META_PATH = "meta.json"
LOG_PATH = "model_log"


def enable_gpu(enable:bool):
    if not enable:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Snore_Dataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = self.__load_features()

    def __load_features(self):
        if not os.path.exists(self.dataset_path):
            raise FileExistsError("Something wrong with dataset directory")
        
        data = [os.path.join(self.dataset_path, el) for el in tqdm(os.listdir(self.dataset_path))]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = np.load(self.data[idx], allow_pickle=True)
        np.random.shuffle(arr)
        features = []
        labels = []
        
        [[features.append(el["features"]), labels.append(el["label"])] for el in arr]
        
        return np.array(features), np.array(labels, dtype=np.int64)
        # return features, label
    
    
class Training:
    def __init__(self, device, **kwargs):
        self.params = munchify(kwargs)
        for item, v in zip(self.params, self.params.values()):
            print(f"k: {item} --- v: {v}")
        
        self.device = device
        
        if not self.params.train_from_ckpt:
            os.makedirs(os.path.dirname(self.params.meta_path), exist_ok=True)
            with open(self.params.meta_path, "w+") as meta:
                json.dump({"params": [kwargs]}, meta, indent=4)

            self.logger = log_init(self.params.model_path, LOG_PATH, "w")
            self.logger.info(
                f"Model '{self.params.model_path}' has been created with these params:")

            self.logger.info(
                f"---------------MODEL AND UTILS SUMMARY---------------")
            # self.logger.info(
            #     f"ENCODER:\n{inspect.getsource(Encoder)}")
            for k, v in kwargs.items():
                self.logger.info(f"{k}: {v}")
            self.logger.info(
                f"-----------------------------------------------------")
        
        else:
            self.logger = log_init(self.params.model_path, LOG_PATH, True, "a+")
            self.logger.info("Params have been loaded successfully for training from checkpoint!")
            
    def auto_train(self):
        self.train()
    
    def train_from_ckpt(self):
        self.train()
    
    @staticmethod
    def accuracy(y_pred, y_true):
        assert y_pred.shape == y_true.shape
        a = torch.sum(y_pred == y_true)
        return a / y_pred.shape[0]
    
    @staticmethod
    def precision(y_pred, y_true):
        assert y_pred.shape == y_true.shape
        tp = torch.sum((y_pred == 1) & (y_true == 1))
        fp = torch.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) 
    
    @staticmethod
    def recall(y_pred, y_true):
        assert y_pred.shape == y_true.shape
        tp = torch.sum((y_pred == 1) & (y_true == 1))
        fn = torch.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) 
        
    def train(self):        
        train_data_dir = os.path.join(self.params.dataset_path, 
                                self.params.features_type,
                                f"sr_{self.params.sample_rate}",
                                f"stride_{self.params.stride}_ms",
                                self.params.train_folder)
        val_data_dir = os.path.join(self.params.dataset_path, 
                                self.params.features_type,
                                f"sr_{self.params.sample_rate}",
                                f"stride_{self.params.stride}_ms",
                                self.params.val_folder)
        if not os.path.exists(train_data_dir) or not os.path.exists(val_data_dir):
            raise Exception("Dataset dirs are not existing!")
        
        training_data = Snore_Dataset(train_data_dir)
        
        train_dataset = DataLoader(training_data, 
                                   batch_size=None, 
                                   shuffle=True, 
                                   drop_last=False, 
                                   pin_memory=True, 
                                   num_workers=6,
                                   persistent_workers=True
                                   )
        
        val_data = Snore_Dataset(val_data_dir)
        
        val_dataset = DataLoader(val_data, 
                                   batch_size=None, 
                                   shuffle=True, 
                                   drop_last=False, 
                                   pin_memory=True, 
                                   num_workers=6,
                                   persistent_workers=True
                                   )

        criterion = nn.CrossEntropyLoss()
            
        model_manager = ModelLoader(feature_type=self.params.features_type, model_type=self.params.model_type, device=self.device)
        
        # optimizer = torch.optim.Adam(model.model.parameters())
        optimizer = torch.optim.AdamW(model_manager.model.parameters(), lr=0.001, amsgrad=True)
        # optimizer = torch.optim.SGD(model.model.parameters(), lr=0.0003, weight_decay=1e-5)
        
        self.start_ckpt_id = 0
        ckpt = Checkpointing(self.params.checkpoint_path,
                             max_to_keep=10,
                             model=model_manager.model,
                             optimizer=optimizer)
        if self.params.train_from_ckpt:
            self.start_ckpt_id = ckpt.load(return_idx=True)
                
        # loss_plot = []
        
        for epoch in range(self.start_ckpt_id, self.params.epochs):
            start = time.time()
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_re = 0
            epoch_pr = 0
            
            val_epoch_loss = 0
            val_epoch_accuracy = 0
            val_epoch_re = 0
            val_epoch_pr = 0

            model_manager.model.train()
            for (batch, (img_tensor, target)) in tqdm(enumerate(train_dataset)):
                img_tensor, target = img_tensor.to(self.device), target.type(torch.int64).to(self.device)
                
                optimizer.zero_grad()
                prediction = model_manager.predict(img_tensor)

                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                
                accuracy = self.accuracy(torch.argmax(prediction, dim=-1), target)
                recall = self.recall(torch.argmax(prediction, dim=-1), target)
                precision = self.precision(torch.argmax(prediction, dim=-1), target)
                
                epoch_loss += loss.item()
                epoch_accuracy += accuracy.item()
                epoch_re += recall.item()
                epoch_pr += precision.item()
                                
                if batch % 100 == 0:
                    self.logger.info(
                    f"Epoch {epoch + 1} Batch {batch} Loss = {loss.item()} --- train_accuracy = {accuracy} --- train_recall = {recall} --- train_precision = {precision}"
                    )
            # storing the epoch end loss value to plot later
            # loss_plot.append(total_loss / num_steps)
            self.logger.info(f"Epoch mean loss = {epoch_loss / batch} --- Epoch mean accuracy = {epoch_accuracy / batch} --- Epoch mean recall = {epoch_re / batch} --- Epoch mean precision = {epoch_pr / batch}")

            model_manager.model.eval()
            with torch.no_grad():
                for (batch, (img_tensor, target)) in tqdm(enumerate(val_dataset)):
                    img_tensor, target = img_tensor.to(self.device), target.type(torch.int64).to(self.device)
                    
                    prediction = model_manager.predict(img_tensor)
                    
                    loss = criterion(prediction, target)
                
                    accuracy = self.accuracy(torch.argmax(prediction, dim=-1), target)
                    recall = self.recall(torch.argmax(prediction, dim=-1), target)
                    precision = self.precision(torch.argmax(prediction, dim=-1), target)
                
                    val_epoch_loss += loss.item()
                    val_epoch_accuracy += accuracy.item()
                    val_epoch_re += recall.item()
                    val_epoch_pr += precision.item()
                
                    if batch % 100 == 0:
                        self.logger.info(
                        f"Validation Epoch {epoch + 1} Batch {batch} Loss = {loss.item()} --- train_accuracy = {accuracy} --- train_recall = {recall} --- train_precision = {precision}"
                        )
                    # storing the epoch end loss value to plot later
                    # loss_plot.append(total_loss / num_steps)
            
            self.logger.info(f"Epoch mean loss = {val_epoch_loss / batch} --- Epoch mean accuracy = {val_epoch_accuracy / batch} --- Epoch mean recall = {val_epoch_re / batch} --- Epoch mean precision = {val_epoch_pr / batch}")


            ckpt.save()
            self.logger.info(f"Epoch {epoch} checkpoint saved!")

            self.logger.info(
                f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

        self.logger.info(f"Training is done!")


class SnoreActivityDetectorLoader:
    loaded = False
    dataset_path = "D:/SnoreFeatures"
    train_folder = "train"
    val_folder = "validation"
    features_type = "mfcc"
    model_type = "CB"
    sample_rate = 16000
    stride = 128
    BATCH_SIZE = 256
    EPOCHS = 100

    def __init__(self, model_name: str, ckpt_idx=None, device='cpu'):
        self.model_path = os.path.join(DIR_PATH,
                                       TRAINED_MODELS_DIR,
                                       model_name)
        
        self.device = device
        self.checkpoint_path = os.path.join(self.model_path, CKPT_DIR)
        self.meta_path = os.path.join(self.model_path, META_PATH)

        if os.path.exists(self.checkpoint_path) and len(os.listdir(self.checkpoint_path)) and os.path.exists(self.meta_path):
            print(f"Model has been found at {self.model_path}")
            self.loaded = True
        elif not os.path.exists(self.model_path):
            print(f"Model will be created at {self.model_path}")
        else:
            if input(f"Model is corrupted. Remove? (y/AB): ") == "y":
                shutil.rmtree(self.model_path)
                self.loaded = False
            else:
                self.loaded = True

        if self.loaded:
            try:
                with open(self.meta_path, "r") as meta:
                    self.params = munchify(json.load(meta)["params"])[0]
            except:
                raise IOError("config file reading error!")
                    
            self.model = ModelLoader(feature_type=self.params.features_type, model_type=self.params.model_type)

            ckpt = Checkpointing(path=self.checkpoint_path,
                                 model=self.model.model)
            ckpt.load(idx=ckpt_idx)

    def input(self, inp, val):
        return val if inp == "" else inp

    def train(self):
        if self.loaded:
            if input("You will loose your model. Proceed? (y/AB): ") == "y":
                shutil.rmtree(self.model_path)
                self.loaded = False
            else:
                return

        if not self.loaded:
            print("\nPlease enter some data for new model: ")
            try:
                self.dataset_path = self.input(
                    input(f"dataset_path - Default = {self.dataset_path}: "), self.dataset_path)
                self.train_folder = self.input(
                    input(f"train_folder - Default = {self.train_folder}: "), self.train_folder)
                self.val_folder = self.input(
                    input(f"val_folder - Default = {self.val_folder}: "), self.val_folder)
                self.features_type = self.input(
                    input(f"dataset features type - Default = {self.features_type}: "), self.features_type)
                self.model_type = self.input(
                    input(f"model type (CB=CNN_BILSTM; SA=SelfAttentive) - Default = {self.model_type}: "), self.model_type)
                self.sample_rate = self.input(
                    input(f"dataset sample rate - Default = {self.sample_rate}: "), self.sample_rate)
                self.stride = self.input(
                    input(f"dataset stride - Default = {self.stride}: "), self.stride)
                self.BATCH_SIZE = int(self.input(
                    input(f"BATCH_SIZE - Default = {self.BATCH_SIZE}: "), self.BATCH_SIZE))
                self.EPOCHS = int(self.input(
                    input(f"EPOCHS - Default = {self.EPOCHS}: "), self.EPOCHS))
            except:
                raise TypeError("Model params initialization failed")

            self.loaded = True

            train = Training(device=self.device,
                             model_path=self.model_path,
                             dataset_path=self.dataset_path,
                             train_folder=self.train_folder,
                             val_folder=self.val_folder,
                             meta_path=self.meta_path,
                             features_type=self.features_type,
                             model_type=self.model_type,
                             sample_rate=self.sample_rate,
                             stride=self.stride,
                             checkpoint_path=self.checkpoint_path,
                             batch_size=self.BATCH_SIZE,
                             epochs=self.EPOCHS,
                             train_from_ckpt=False)

            train.train()
            
    def train_from_ckpt(self):
        if not self.loaded:
            raise ValueError("Model is not loaded!")
        train = Training(device=self.device,
                         model_path=self.model_path,
                         train_dataset_path=self.dataset_path,
                         val_dataset_path=self.val_dataset_path,
                         checkpoint_path=self.checkpoint_path,
                         features_type=self.params.features_type,
                         model_type=self.params.model_type,
                         sample_rate=self.sample_rate,
                         stride=self.stride,
                         meta_path=self.meta_path,
                         batch_size=self.params.batch_size,
                         epochs=self.params.epochs,
                         train_from_ckpt=True)

        train.train()
    
    def save_ckpt_as_model(self, name, output_dir, meta_file_name="meta.json"):
        working_dir = os.path.join(output_dir, name)
        os.makedirs(working_dir, exist_ok=True)
        
        torch.save(self.model.model.state_dict(), os.path.join(working_dir, f"{name}.pt"))
        with open(os.path.join(working_dir, meta_file_name), "w+") as meta:
            json.dump({ "features_type": self.params.features_type, 
                        "model_type": self.params.model_type,
                        "sample_rate": int(self.params.sample_rate)
                      }, meta, indent=4)
            
    def predict(self):
        pass
     
     
if __name__ == "__main__": 
    device = enable_gpu(True)
    sad = SnoreActivityDetectorLoader("CA", device=device)
    sad.train()
    # vad.train_from_ckpt()
    # vad.save_ckpt_as_model("SoundNet", "C:\\Users\\shace\\Documents\\GitLab\\vad\\VAD\\inference_models")