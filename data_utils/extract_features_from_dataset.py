import argparse
import json
import multiprocessing
import os
import shutil
from collections import defaultdict
from itertools import chain
from os.path import join
from random import choice

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from munch import munchify, Munch
from tqdm import tqdm

from feature_extractors.main_fe import MFCC_FE, MelSpec_FE, Raw_FE
from augmentations import (
    AugSequence,
    Gain,
    Identity,
    Impulse,
    LowPassFilter,
    TimeShift,
    TrimSilence,
    EqualizeAmplitude,
    BackgroundNoise,
    GaussianNoise,
)
from multiprocess_funcs import compute_threads_work


CONFIG_DUMP_NAME = "extraction_config.yaml"

GENERAL_PARAMS = "general"
FEATURE_TYPES = {"mel": MelSpec_FE, "mfcc": MFCC_FE, "raw": Raw_FE}

AUGMENTATIONS = {
    "equalize_amplitude": EqualizeAmplitude,
    "background_noise": BackgroundNoise,
    "gaussian_noise": GaussianNoise,
    "random_gain": Gain,
    "time_shift": TimeShift,
    "impulse": Impulse,
    "low_pass_filter": LowPassFilter,
    "trim_silence": TrimSilence,
}


class FeatureExtractor:
    def __init__(self, config_path):
        with open(config_path) as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        self.config = config

    def ms2samples(self, ms: int, sample_rate: int):
        """
        Converts milliseconds to samples with desired sample rate

        Args:
            ms (int): milliseconds to convert
            sample_rate (int): desired sample rate

        Returns:
            int: samples
        """
        return int(ms * sample_rate / 1000)

    def sec2ms(self, sec: float):
        """
        Converts seconds to milliseconds

        Args:
            sec (float): seconds to convert

        Returns:
            int: milliseconds
        """
        return int(1000 * sec)

    def ms2sec(self, ms: int):
        """
        Converts milliseconds to seconds

        Args:
            ms (int): milliseconds to convert

        Returns:
            float: seconds
        """
        return ms / 1000

    def range_generator(self, start: int, end: int):
        """
        Generates a sequence of start and end indices based on
        the given start and end values, sample size, and stride.

        Args:
            start (int): The starting index of the range.
            end (int): The ending index of the range.

        Yields:
            tuple: A tuple containing the start and end
            indices for each interval in the range.
        """
        cnt = (end - start - self.samples_per_square) // self.stride
        for k in range(cnt + 1):
            yield (s := start + self.stride * k), s + self.samples_per_square

    def extractor(self, meta_info, source_audio):
        """
        Process audio data by applying
        augmentations and extracting features.

        Args:
            meta_info (Munch): Information about the audio data,
            including the label and augmentations to apply.
            source_audio (torch.Tensor): The raw audio data to process.

        Returns:
            None: The processed features
            are saved as numpy arrays.
        """
        rng = self.range_generator(0, source_audio.shape[-1])
        for s, e in rng:
            if self.limiter and self.f_cnt[meta_info.label] >= self.limiter:
                return

            interval = source_audio[:, s:e]
            augm_interval = meta_info.augs(torch.unsqueeze(interval, 0))
            augm_interval = torch.squeeze(augm_interval, 0).cpu()

            if torch.sum(torch.isnan(augm_interval)):
                return

            features = self.transform(augm_interval).numpy().astype(np.float32)

            self.feat_buff.append((features, self.training_labels[meta_info.label]))

            self.f_cnt[meta_info.label] += 1

            if len(self.feat_buff) == self.main_cfg.batch_size:
                batch_file_name = f"batch_{os.getpid()}_{self.f_c}.npy"
                np.save(
                    join(self.features_dir, batch_file_name),
                    np.array(
                        [{"features": l[0], "label": l[-1]} for l in self.feat_buff]
                    ),
                )

                self.f_c += 1
                self.feat_buff.clear()

    def file_load_and_proceed(self, label_file: list, start: int, stop: int):
        """
        Load audio files, apply transformations,
        and extract features from the audio data.

        Args:
            label_file (list): A list of labeled audio files.
            start (int): The starting index of
            the subset of label_file to process.
            stop (int): The ending index of
            the subset of label_file to process.

        Returns:
            None: The processed features are saved as numpy arrays.
        """
        curr_label_file = label_file[start:stop]
        np.random.shuffle(curr_label_file)

        self.f_cnt = {k: 0 for k in self.training_labels}

        self.limiter = 0
        if "limit_for_class" in self.main_cfg:
            self.limiter = self.main_cfg.limit_for_class // self.threads

        for rec in tqdm(curr_label_file):
            if self.limiter and self.f_cnt[rec.label] >= self.limiter:
                continue

            source_audio, sample_rate = torchaudio.load(rec.audio_path)
            source_audio = source_audio.to(self.device)

            if source_audio.shape[-1] < self.samples_per_square:
                continue

            source_audio = (
                torch.mean(source_audio, dim=0, keepdim=True)
                if source_audio.shape[0] > 1
                else source_audio
            )

            if sample_rate != self.main_cfg.sample_rate:
                transform = torchaudio.transforms.Resample(
                    sample_rate, self.main_cfg.sample_rate
                ).to(self.device)
                source_audio = transform(source_audio)

            s = self.ms2samples(self.sec2ms(rec.start), self.main_cfg.sample_rate)
            e = self.ms2samples(self.sec2ms(rec.end), self.main_cfg.sample_rate)

            e = source_audio.shape[-1] if e > source_audio.shape[-1] else e

            self.extractor(rec, source_audio[:, s:e])

        if len(self.feat_buff):
            batch_file_name = f"part_batch_{os.getpid()}_{self.f_c}.npy"
            np.save(
                join(self.features_dir, batch_file_name),
                np.array([{"features": l[0], "label": l[-1]} for l in self.feat_buff]),
            )
            self.feat_buff.clear()

    def part_batch_combiner(self):
        """
        Combines partial batches of feature data
        into larger batches for further processing.

        Returns:
            None: The partial batches are combined into
            larger batches and saved as separate files.
        """
        file_list = filter(lambda x: "part" in x, os.listdir(self.features_dir))
        batch_buff = []
        for idx, file in enumerate(file_list):
            file_path = join(self.features_dir, file)

            batch_sum = sum([el.shape[0] for el in batch_buff])
            arr = np.load(file_path, allow_pickle=True)
            if batch_sum + arr.shape[0] <= self.main_cfg.batch_size:
                batch_buff.append(arr)
            else:
                batch_buff.append(arr[: self.main_cfg.batch_size - batch_sum])
                np.save(
                    join(self.features_dir, f"combined_batch_{idx}.npy"),
                    np.concatenate(batch_buff),
                )
                batch_buff.clear()
                batch_buff.append(arr[self.main_cfg.batch_size - batch_sum :])

            os.remove(file_path)
        if len(batch_buff):
            np.save(
                join(self.features_dir, f"last_combined_batch_{idx}.npy"),
                np.concatenate(batch_buff),
            )

    def shuffle_batches(self):
        """
        Shuffles batches of feature data stored
        as numpy arrays in a directory.
        """
        files = os.listdir(self.features_dir)
        np.random.shuffle(files)

        f_idx = 0
        t_idx = self.main_cfg.shuffle_window

        while f_idx < len(files):
            window = files[f_idx:t_idx]

            window_data = [
                np.load(join(self.features_dir, batch), allow_pickle=True)
                for batch in window
            ]
            data_len = {
                file_name: el.shape[0] for file_name, el in zip(window, window_data)
            }

            window_data = np.concatenate(window_data)
            window_data = window_data[np.random.permutation(window_data.shape[0])]

            tmp_f_idx = 0
            tmp_t_idx = 0
            upd_files = {}
            for k, v in data_len.items():
                tmp_f_idx = tmp_t_idx
                tmp_t_idx += v
                upd_files.update({k: np.array(window_data[tmp_f_idx:tmp_t_idx])})

            [
                np.save(join(self.features_dir, el), feat)
                for el, feat in upd_files.items()
            ]

            print(f"{t_idx} of {len(files)} files shuffled")

            f_idx += self.main_cfg.shuffle_hop
            t_idx += self.main_cfg.shuffle_hop
            print("-----------------------")

    def get_augsequence(self, augs: list):
        """
        Returns an instance of the AugSequence class,
        which applies a sequence of augmentations to audio data.

        Args:
            augs (list): A list of augmentation
            names to be applied to the audio data.

        Returns:
            AugSequence instance: An instance of the AugSequence class,
            which can be used to apply the specified augmentations to audio data.
        """
        return AugSequence(
            [
                AUGMENTATIONS[aug](
                    **self.main_cfg.augmentations_config[aug], device=self.device
                )
                for aug in augs
            ]
            if len(augs)
            else [Identity(probability=1.0)]
        ).to(self.device)

    def is_between(self, val):
        for k, v in self.dir_map.items():
            if v[0] <= val < v[1]:
                return k

    def train_validation_test_split(self):
        """
        Split the feature files into three directories:
        train, validation, and test.

        Returns:
            None. The files are moved into the
            train, validation, and test directories.
        """
        train_dir = os.path.join(self.features_dir, self.main_cfg.train_folder)
        val_dir = os.path.join(self.features_dir, self.main_cfg.val_folder)
        test_dir = os.path.join(self.features_dir, self.main_cfg.test_folder)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        files = map(
            lambda x: os.path.join(self.features_dir, x), os.listdir(self.features_dir)
        )
        files = list(filter(os.path.isfile, files))

        train_ratio = int(len(files) * self.main_cfg.train_val_split[0] / 100)
        val_ratio = (
            int(len(files) * self.main_cfg.train_val_split[1] / 100) + train_ratio
        )
        test_ratio = len(files)
        self.dir_map = {
            train_dir: (0, train_ratio),
            val_dir: (train_ratio, val_ratio),
            test_dir: (val_ratio, test_ratio),
        }

        np.random.shuffle(files)
        [shutil.move(el, self.is_between(idx)) for idx, el in enumerate(files)]

    def audio_chunks_amount(self, duration: int):
        """
        Calculate the number of audio chunks
        that can be extracted from a given duration.

        Args:
            duration (int): The duration of
            the audio in milliseconds.

        Returns:
            float: The number of audio chunks
            that can be extracted from the given duration.
        """
        return (duration - self.main_cfg.chunk_size // 1000) // (
            self.main_cfg.stride / 1000
        ) + 1

    def extract_features(self):
        """
        Main extraction function:
        1) Processes the configuration data and creates a dataset of audio
        samples with their corresponding labels and augmentation settings.
        2) Performs various operations to extract features from audio data,
        including counting the number of chunks for each label,
        adding silence samples to the dataset, shuffling the dataset,
        setting up feature extraction parameters,
        and running feature extraction in parallel using multiple threads
        """

        def get_munch_audio(audio_path, label, start, end, augs=None):
            return Munch(
                audio_path=audio_path,
                label=self.labels_merging.get(label, label),
                start=start,
                end=end,
                augs=self.get_augsequence(augs) if augs else self.augm_dict[label],
            )

        self.main_cfg = munchify(self.config[GENERAL_PARAMS])
        self.device = "cuda" if self.main_cfg.use_gpu else "cpu"

        data_keys = list(self.config.keys())
        data_keys.remove(GENERAL_PARAMS)
        if not len(data_keys):
            raise ValueError("No data points in config")

        dataset = []
        for data_p in data_keys:
            data_info = munchify(self.config[data_p])

            if not os.path.exists(data_info.path):
                raise OSError(f"{data_info.path} directory doesn't exist")

            if not os.path.exists(data_info.path) or not len(
                os.listdir(data_info.path)
            ):
                raise OSError(f"Data directory does not exits or empty for {data_p}")
            dir_content = os.listdir(data_info.path)

            if len(data_info.augmentations) != len(data_info.tokens):
                raise Exception(
                    f"Every token must have it's own augmentations list (empty if not used)"
                )

            if (
                len(
                    set(chain(*filter(len, data_info.augmentations)))
                    - set(AUGMENTATIONS)
                )
                > 0
            ):
                raise Exception(f"Some wrong augmentations are specified")

            if "use_equalization" in self.main_cfg:
                self.main_cfg.augmentations_config.equalize_amplitude = Munch(
                    target_dBFS=self.main_cfg.use_equalization, probability=1.0
                )
                [el.append("equalize_amplitude") for el in data_info.augmentations]

            self.augm_dict = {
                k: self.get_augsequence(v)
                for k, v in zip(data_info.tokens, data_info.augmentations)
            }

            if "label_file" in data_info and "audio_folder" in data_info:
                if (
                    not data_info.label_file in dir_content
                    or not data_info.audio_folder in dir_content
                ):
                    raise OSError(f"No label file or audio directory for {data_p}")

                try:
                    t_dataset = json.load(
                        open(join(data_info.path, data_info.label_file))
                    )
                    t_dataset = {
                        k: list(
                            filter(lambda x: any(el in x for el in data_info.tokens), v)
                        )
                        for k, v in t_dataset.items()
                    }
                    t_dataset = {k: v for k, v in t_dataset.items() if v}

                    if not len(t_dataset):
                        raise Exception("Wrong labels are specified")

                    self.labels_merging = {token: token for token in data_info.tokens}
                    if "labels_merging" in self.main_cfg:
                        self.labels_merging.update(self.main_cfg.labels_merging)

                    for audio_name, labels in t_dataset.items():
                        dataset.extend(
                            [
                                get_munch_audio(
                                    audio_path=os.path.join(
                                        data_info.path,
                                        data_info.audio_folder,
                                        audio_name,
                                    ),
                                    label=l[0],
                                    start=l[1][0],
                                    end=l[1][1],
                                )
                                for l in labels
                                if self.sec2ms(l[1][1] - l[1][0])
                                >= self.main_cfg.chunk_size
                            ]
                        )

                except Exception as e:
                    raise e

            else:
                if len(data_info.tokens) > 1:
                    raise Exception("Only one token for not labeled data is available")

                self.labels_merging = {token: token for token in data_info.tokens}
                if "labels_merging" in self.main_cfg:
                    self.labels_merging.update(self.main_cfg.labels_merging)

                for audio in os.listdir(data_info.path):
                    audio_path = os.path.join(data_info.path, audio)
                    audio = sf.SoundFile(audio_path)

                    duration = np.round(len(audio) / audio.samplerate, 1)
                    if self.sec2ms(duration) >= self.main_cfg.chunk_size:
                        dataset.append(
                            get_munch_audio(
                                audio_path=audio_path,
                                label=data_info.tokens[0],
                                start=0.0,
                                end=duration,
                            )
                        )

        chunks_counter = defaultdict(int)
        for el in dataset:
            chunks_counter[el.label] += self.audio_chunks_amount(el.end - el.start)

        if "use_silence" in self.main_cfg:
            noises = os.listdir(self.main_cfg.use_silence.silence_path)
            if not noises:
                raise OSError("Noise examples directory is empty")

            [
                dataset.append(
                    get_munch_audio(
                        audio_path=join(
                            self.main_cfg.use_silence.silence_path, choice(noises)
                        ),
                        label="__silence__",
                        start=0.0,
                        end=self.ms2sec(self.main_cfg.chunk_size),
                        augs=self.main_cfg.use_silence.augmentations,
                    )
                )
                for _ in range(
                    int(
                        self.main_cfg.use_silence.ratio
                        * chunks_counter[self.main_cfg.training_tokens[-1]]
                    )
                )
            ]

            self.main_cfg.training_tokens.insert(0, "__silence__")

        tokens_diff = set(chunks_counter.keys()) - set(self.main_cfg.training_tokens)
        if tokens_diff:
            raise ValueError(
                f"There are tokens in data, that are not \
                presented in training tokens:\n{tokens_diff}"
            )

        self.training_labels = {
            v: idx for idx, v in enumerate(self.main_cfg.training_tokens)
        }

        np.random.shuffle(dataset)

        self.features_dir = self.main_cfg.output_path

        os.makedirs(self.features_dir, exist_ok=False)

        self.samples_per_square = self.ms2samples(
            self.main_cfg.chunk_size, self.main_cfg.sample_rate
        )
        self.win_len = self.ms2samples(self.main_cfg.win_len, self.main_cfg.sample_rate)
        self.n_fft = self.main_cfg.n_fft
        self.hop_len = self.ms2samples(self.main_cfg.hop_len, self.main_cfg.sample_rate)

        self.stride = self.ms2samples(self.main_cfg.stride, self.main_cfg.sample_rate)

        self.feat_buff = []
        self.f_c = 0

        self.transform = FEATURE_TYPES[self.main_cfg.feature_type](
            sample_rate=self.main_cfg.sample_rate,
            win_length=self.win_len,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            n_mels=self.main_cfg.n_mels,
        )

        if "limit_for_class" in self.main_cfg:
            if isinstance(self.main_cfg.limit_for_class, str):
                self.main_cfg.limit_for_class = chunks_counter[
                    self.main_cfg.limit_for_class
                ]
            elif not isinstance(self.main_cfg.limit_for_class, int):
                raise ValueError(
                    "Wrong limitation type. \
                    Limiting with one of the labels or with int are only possibilities"
                )

        self.threads = self.main_cfg.compute_threads if not self.main_cfg.use_gpu else 1
        multiprocessing.set_start_method("spawn")
        threads = [
            multiprocessing.Process(
                target=self.file_load_and_proceed, args=(dataset, arg[0], arg[1])
            )
            for arg in compute_threads_work(len(dataset), self.threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.part_batch_combiner()
        self.shuffle_batches()
        if (
            all(0 < el < 100 for el in self.main_cfg.train_val_split)
            and sum(self.main_cfg.train_val_split) < 100
        ):
            self.train_validation_test_split()

        with open(join(self.features_dir, CONFIG_DUMP_NAME), "w+") as dmp:
            yaml.dump(self.config, dmp)

        print("Extraction successfully completed!")


# python3 extract_features_from_dataset.py --config-path features_extraction_configs/extractor_config.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path", type=str, help="Path to conf file", required=True
    )

    args = parser.parse_args()

    feature_extractor = FeatureExtractor(args.config)
    feature_extractor.extract_features()
