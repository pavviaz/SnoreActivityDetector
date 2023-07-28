import argparse
import json
import multiprocessing
import os
import numpy as np
import torchaudio
from tqdm import tqdm
from feature_extractors import MelSpec_FE, MFCC_FE, Raw_FE 
from random import shuffle
from os.path import join
import shutil


LABEL_FILE = "labels.json"
AUDIO_FOLDER = "audios"

WIN_LEN = 10  # ms
HOP_LEN = 30  # ms
MEL_SIZE = 2000  # ms

FEATURE_TYPES = {"mel": MelSpec_FE,
                 "mfcc": MFCC_FE,
                 "raw": Raw_FE}


class FeatureExtractor:
    def __init__(self, argparse):
        self.__dict__.update(vars(argparse))  # make argparser args accessible with self.
    
    def ms2samples(self, ms, sample_rate):
        return int(ms * sample_rate / 1000)

    def sec2ms(self, sec):
        return int(1000 * sec)
    
    def compute_threads_work(self, length, working_threads):
        amount = length // working_threads
        while length > 0:
            tmp = length
            length = 0 if length - amount < 0 else length - amount
            yield (length, tmp)
    
    def range_generator(self, start, end):
        cnt = (end - start - self.samples_per_square) // self.stride
        for k in range(0, cnt + 1):
            yield (s := start + self.stride * k), s + self.samples_per_square  # python 3.8+
            
    def extractor(self, meta_info, source_audio):
        start, end = self.ms2samples(self.sec2ms(meta_info[1][0]), self.sample_rate), self.ms2samples(self.sec2ms(meta_info[1][1]), self.sample_rate)
        end = source_audio.shape[-1] if end > source_audio.shape[-1] else end

        rng = self.range_generator(start, end)
        for el in rng:
            interval = source_audio[:, el[0]: el[1]]
            
            features = self.transform(interval).numpy()
            self.feat_buff.append((features, self.training_labels[meta_info[0]]))
            
            if len(self.feat_buff) == self.batch_size:
                batch_file_name = f"batch_{os.getpid()}_{self.f_c}.npy"
                np.save(join(self.features_dir, 
                             batch_file_name), 
                        np.array([{"features": l[0], "label": l[-1]} for l in self.feat_buff]))
                
                self.f_c += 1
                self.feat_buff.clear()

    def file_load_and_proceed(self, label_file, start, stop):
        curr_label_file = label_file[start: stop]
        for pair in tqdm(curr_label_file):
            for file_name, label in pair.items():
                source_audio, sample_rate = torchaudio.load(join(self.dataset_path, 
                                                                 AUDIO_FOLDER, 
                                                                 file_name))
                
                if sample_rate != self.sample_rate:
                    transform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                    source_audio = transform(source_audio)

                [self.extractor(meta, source_audio) for meta in label]
                
                if len(self.feat_buff):
                    batch_file_name = f"part_batch_{os.getpid()}_{self.f_c}.npy"
                    np.save(join(self.features_dir, 
                                 batch_file_name), 
                        np.array([{"features": l[0], "label": l[-1]} for l in self.feat_buff]))

    def part_batch_combiner(self):
        file_list = filter(lambda x: "part" in x, os.listdir(self.features_dir))
        batch_buff = []
        for idx, file in enumerate(file_list):
            file_path = join(self.features_dir, file)
            
            batch_sum = sum([el.shape[0] for el in batch_buff])
            arr = np.load(file_path, allow_pickle=True)
            if batch_sum + arr.shape[0] <= self.batch_size:
                batch_buff.append(arr)
            else:
                batch_buff.append(arr[:self.batch_size - batch_sum])
                np.save(join(self.features_dir, 
                             f"combined_batch_{idx}.npy"), 
                        np.concatenate(batch_buff))
                batch_buff.clear()
                batch_buff.append(arr[self.batch_size - batch_sum:])
                
            os.remove(file_path)
        if len(batch_buff):
            np.save(join(self.features_dir, 
                         f"last_combined_batch_{idx}.npy"), 
                    np.concatenate(batch_buff))
    
    def shuffle_batches(self):
        files = os.listdir(self.features_dir)
        np.random.shuffle(files)
        
        f_idx = 0
        t_idx = self.shuffle_window
        
        while f_idx < len(files):
            window = files[f_idx: t_idx]
                        
            window_data = [np.load(join(self.features_dir, batch), allow_pickle=True) for batch in window]
            data_len = {file_name: el.shape[0] for file_name, el in zip(window, window_data)}
            
            window_data = np.concatenate(window_data)
            window_data = window_data[np.random.permutation(window_data.shape[0])]
            
            tmp_f_idx = 0
            tmp_t_idx = 0
            upd_files = {}
            for k, v in data_len.items():
                tmp_f_idx = tmp_t_idx
                tmp_t_idx += v
                upd_files.update({k: np.array(window_data[tmp_f_idx: tmp_t_idx])})
            
            [np.save(join(self.features_dir, el), feat) for el, feat in upd_files.items()]
            
            print(f"{t_idx} of {len(files)} files shuffled")
            
            f_idx += self.shuffle_hop
            t_idx += self.shuffle_hop
            print("-----------------------")

    def train_validation_split(self):
        train_dir = os.path.join(self.features_dir, self.train_folder)
        val_dir = os.path.join(self.features_dir, self.val_folder)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        files = map(lambda x: os.path.join(self.features_dir, x), os.listdir(self.features_dir))
        files = list(filter(os.path.isfile, files))
        train_ratio = int(len(files) * self.train_val_split / 100)

        np.random.shuffle(files)
        [shutil.move(el, train_dir) if idx < train_ratio else shutil.move(el, val_dir) for idx, el in enumerate(files)]
        
    def extract_features(self):
        if not os.path.exists(self.dataset_path):
            raise Exception(f"{self.dataset_path} directory doesn't exist ")
        dir_content = os.listdir(self.dataset_path)
        if not LABEL_FILE in dir_content or not AUDIO_FOLDER in dir_content:
            raise Exception("Something wrong with dataset directory")
        try:
            dataset = json.load(open(join(self.dataset_path, LABEL_FILE), "r"))
            dataset = {k: list(filter(lambda x: any(el in x for el in self.training_tokens), v)) for k, v in dataset.items()}
            dataset = {k: v for k, v in dataset.items() if len(v)}

            if not len(dataset):
                raise Exception("Wrong labels are specified")

            dataset = [{item[0]: item[1]} for item in dataset.items()]
            shuffle(dataset)
            if self.limit_length:
                dataset = dataset[:self.limit_length]
        except Exception as e:
            raise e
        
        self.training_labels = {v: idx for idx, v in enumerate(self.training_tokens)}

        self.features_dir = join(self.output_path, 
                                 self.feature_type, 
                                 f"sr_{self.sample_rate}", 
                                 f"stride_{self.stride}_ms")

        os.makedirs(self.features_dir, exist_ok=True)
        
        self.samples_per_square = self.ms2samples(MEL_SIZE, self.sample_rate)
        self.win_len = self.ms2samples(WIN_LEN, self.sample_rate)
        self.hop_len = self.ms2samples(HOP_LEN, self.sample_rate)
        
        self.stride = self.ms2samples(self.stride, self.sample_rate)

        self.feat_buff = []
        self.f_c = 0

        self.transform = FEATURE_TYPES[self.feature_type](sample_rate=self.sample_rate,
                                                          win_length=self.win_len,
                                                          n_fft=self.win_len,
                                                          hop_length=self.hop_len,
                                                          n_mels=40)
        
        multiprocessing.set_start_method('spawn')
        threads = [multiprocessing.Process(target=self.file_load_and_proceed, args=(dataset, arg[0], arg[1])) for arg in
                self.compute_threads_work(len(dataset), self.compute_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
                
        self.part_batch_combiner()
        self.shuffle_batches()
        if 0 < self.train_val_split < 100:
            self.train_validation_split()

        print("Extraction successfully completed!")


# python3 extract_features_from_dataset.py --dataset-path data --feature-type mfcc --output-path snore_features --stride 32 --sample-rate 16000 --compute-threads 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the audio dataset directory. \
        It must contain subfolder named 'audios' with audio file in '.wav' format in it \
        'and label.json' file with label format like \
        { 'audio_name': [['<SPEECH_TOKEN>', [0.0, 5.67]]], [['<NO_SPEECH_TOKEN>', [5.67, 7.4]]], [], ... }. \
        You can specify <SPEECH_TOKEN> and <NO_SPEECH_TOKEN> as optional param, 'SPEECH' and 'NO_SPEECH'  by default",
        required=True
    )
    
    # parser.add_argument(
    #     "--snore-token",
    #     help="A token indicating the presence of snore on the audio \
    #     'SNORE' by default",
    #     type=str,
    #     default="SNORE"
    # )

    # parser.add_argument(
    #     "--no-snore-token",
    #     help="A token indicating the absence of snore on the audio \
    #     'NO_SNORE' by default",
    #     type=str,
    #     default="NO_SNORE"
    # )

    parser.add_argument(
        "--training-tokens",
        help="Tokens that will be used during training (whitespace separated). \
        Note, only tokens presented in label file are allowed",
        type=str,
        nargs='+',
        required=True
    )
    
    parser.add_argument(
        "--stride",
        help="Controls frame step during label generation. \
        For example, if this param equals 32 (default value), \
        start values of generated label wii be (0, 320), (32, 352), ...",
        type=int,
        default=32
    )
    
    parser.add_argument(
        "--sample-rate",
        help="Sample rate of the dataset. \
        If some audios in the dataset have different SR, \
        they will be transformed to the target one",
        type=int,
        default=16000
    )

    parser.add_argument(
        "--feature-type",
        help=f"Following feature types are supported: {', '.join(FEATURE_TYPES.keys())}",
        required=True
    )
    
    parser.add_argument(
        "--limit-length",
        help=f"Sets audio files amount being used for feature extraction. No limit by default",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--compute-threads",
        help="Number of feature computing threads. \
              If this param equals 1 (default value), multiprocessing will not be applied",
        type=int,
        default=1
    )
    
    parser.add_argument(
        "--batch-size",
        help="Size of features stored in one file. \
              If this param equals 1 (default value), multiprocessing will not be applied",
        type=int,
        default=256
    )
    
    parser.add_argument(
        "--shuffle-window",
        help="A value, representing the number of elements \
              from dataset which will be shuffled simultaneously. \
              The bigger this value is, the better shuffling is being achieved",
        type=int,
        default=100
    )
    
    parser.add_argument(
        "--shuffle-hop",
        help="Shuffling window step. \
        The lesser this value is, the better shuffling is being achieved",
        type=int,
        default=5
    )

    parser.add_argument(
        "--output-path",
        help="Output directory for extracted features. By default is the same as dataset-path \
              Subfolder named as feature type will be created at this directory, \
              containing batched '.npy' files named \
              '{batch}_{extracting-process-id}_{index}', for example 'batch_34604_0.npy'",
        type=str,
        required=False
    )

    parser.add_argument(
        "--train-val-split",
        help="Splits data to train and validation folders in specified ratio. \
              Given value (in percent) of data will be train set and the rest is for validation as well. \
              Works only if 0 < specified value < 100",
        type=int,
        default=80
    )

    parser.add_argument(
        "--train-folder",
        help="Name of folder for train set",
        type=str,
        default="train"
    )

    parser.add_argument(
        "--val-folder",
        help="Name of folder for validation set",
        type=str,
        default="validation"
    )

    args = parser.parse_args()
    
    if not args.output_path:
        args.output_path = args.dataset_path
        
    feature_extractor = FeatureExtractor(args)
    feature_extractor.extract_features()
