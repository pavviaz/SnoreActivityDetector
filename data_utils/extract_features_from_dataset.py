import argparse
import json
import multiprocessing
import os
import shutil
from collections import ChainMap
from itertools import chain
from os.path import join

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from munch import munchify
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


class ZipIterator:
    def __init__(self, list_1=[], list_2=[]):
        assert len(list_1) == len(list_1)

        self.list_1 = list_1
        self.list_2 = list_2
        self.current = -1

    def update(self, list_1, list_2):
        assert len(list_1) == len(list_1)

        self.list_1.extend(list_1)
        self.list_2.extend(list_2)

    def __len__(self):
        return len(self.list_1)

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < len(self.list_1):
            return self.list_1[self.current], self.list_2[self.current]
        raise StopIteration


class FeatureExtractor:
    def __init__(self, config_path):
        with open(config_path) as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        self.config = config

    def ms2samples(self, ms, sample_rate):
        return int(ms * sample_rate / 1000)

    def sec2ms(self, sec):
        return int(1000 * sec)

    def range_generator(self, start, end):
        cnt = (end - start - self.samples_per_square) // self.stride
        for k in range(cnt + 1):
            yield (
                s := start + self.stride * k
            ), s + self.samples_per_square  # python 3.8+

    def extractor(self, meta_info, source_audio):
        if "trim_silence" in meta_info.augs:
            meta_info.augs.remove("trim_silence")
            trimmer = self.get_augsequence(["trim_silence"])
            source_audio = torch.squeeze(trimmer(source_audio), 0)

        start, end = 0, source_audio.shape[-1]

        rng = self.range_generator(start, end)
        for fr, to in rng:
            if self.general_params.limit_for_class:
                if hasattr(self, "limiter"):
                    if self.f_cnt[meta_info.label] >= self.limiter:
                        return
                elif (
                    meta_info.label != self.general_params.limit_for_class
                    and self.f_cnt[meta_info.label]
                    >= self.f_cnt[self.general_params.limit_for_class]
                ):
                    return
            interval = source_audio[:, fr:to]
            augs = self.get_augsequence(meta_info.augs)
            augm_interval = augs(torch.unsqueeze(interval, 0))
            augm_interval = torch.squeeze(augm_interval, 0).cpu()

            if torch.sum(torch.isnan(augm_interval)):
                return

            try:
                features = self.transform(augm_interval).numpy().astype(np.float32)
            except Exception as e:
                print(meta_info, end)
                raise e
            self.feat_buff.append((features, self.training_labels[meta_info.label]))

            self.f_cnt[meta_info.label] += 1

            if len(self.feat_buff) == self.general_params.batch_size:
                batch_file_name = f"batch_{os.getpid()}_{self.f_c}.npy"
                np.save(
                    join(self.features_dir, batch_file_name),
                    np.array(
                        [{"features": l[0], "label": l[-1]} for l in self.feat_buff]
                    ),
                )

                self.f_c += 1
                self.feat_buff.clear()

    def match_target_amplitude(self, sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    def file_load_and_proceed(self, label_file, start, stop):
        curr_label_file = label_file[start:stop]
        np.random.shuffle(curr_label_file)
        curr_label_file = dict(ChainMap(*curr_label_file))

        data_iterator = ZipIterator()
        self.f_cnt = {k: 0 for k in self.training_labels}
        if self.general_params.limit_for_class:
            if isinstance(self.general_params.limit_for_class, int):
                self.limiter = self.general_params.limit_for_class // self.threads

            elif isinstance(self.general_params.limit_for_class, str):
                for k, v in curr_label_file.items():
                    strict_r = list(
                        filter(
                            lambda x: x.label == self.general_params.limit_for_class, v
                        )
                    )
                    if len(strict_r):
                        [v.remove(el) for el in strict_r]
                        data_iterator.update([k] * len(strict_r), strict_r)

                curr_label_file = {k: v for k, v in curr_label_file.items() if v}
            else:
                raise Exception(
                    "Wrong limitation type. \
                                Limiting with one of the labels or with int are only possibilities"
                )

        [data_iterator.update([k] * len(v), v) for k, v in curr_label_file.items()]

        for path, meta in tqdm(data_iterator):
            if hasattr(self, "limiter"):
                if self.f_cnt[meta.label] >= self.limiter:
                    continue
            elif (
                meta.label != self.general_params.limit_for_class
                and self.f_cnt[meta.label]
                >= self.f_cnt[self.general_params.limit_for_class]
            ):
                continue

            if not "__silence__" in path:
                source_audio, sample_rate = torchaudio.load(path)
            else:
                # source_audio, sample_rate = torchaudio.load(
                #     r"C:\Users\shace\Documents\GitHub\snore_detector\SnoreActivityDetector\data_utils\tests\greek_mic_noise.wav"
                # )
                source_audio, sample_rate = torch.zeros(1, self.general_params.stride), self.general_params.sample_rate
            source_audio = source_audio.to(self.device)

            if source_audio.shape[-1] < self.samples_per_square:
                continue

            source_audio = (
                torch.mean(source_audio, dim=0, keepdim=True)
                if source_audio.shape[0] > 1
                else source_audio
            )

            if sample_rate != self.general_params.sample_rate:
                transform = torchaudio.transforms.Resample(
                    sample_rate, self.general_params.sample_rate
                ).to(self.device)
                source_audio = transform(source_audio)

            f, t = [
                self.ms2samples(self.sec2ms(el), self.general_params.sample_rate)
                for el in meta.from_to
            ]
            t = source_audio.shape[-1] if t > source_audio.shape[-1] else t

            self.extractor(meta, source_audio[:, f:t])

        if len(self.feat_buff):
            batch_file_name = f"part_batch_{os.getpid()}_{self.f_c}.npy"
            np.save(
                join(self.features_dir, batch_file_name),
                np.array([{"features": l[0], "label": l[-1]} for l in self.feat_buff]),
            )
            self.feat_buff.clear()

    def part_batch_combiner(self):
        file_list = filter(lambda x: "part" in x, os.listdir(self.features_dir))
        batch_buff = []
        for idx, file in enumerate(file_list):
            file_path = join(self.features_dir, file)

            batch_sum = sum([el.shape[0] for el in batch_buff])
            arr = np.load(file_path, allow_pickle=True)
            if batch_sum + arr.shape[0] <= self.general_params.batch_size:
                batch_buff.append(arr)
            else:
                batch_buff.append(arr[: self.general_params.batch_size - batch_sum])
                np.save(
                    join(self.features_dir, f"combined_batch_{idx}.npy"),
                    np.concatenate(batch_buff),
                )
                batch_buff.clear()
                batch_buff.append(arr[self.general_params.batch_size - batch_sum :])

            os.remove(file_path)
        if len(batch_buff):
            np.save(
                join(self.features_dir, f"last_combined_batch_{idx}.npy"),
                np.concatenate(batch_buff),
            )

    def shuffle_batches(self):
        files = os.listdir(self.features_dir)
        np.random.shuffle(files)

        f_idx = 0
        t_idx = self.general_params.shuffle_window

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

            f_idx += self.general_params.shuffle_hop
            t_idx += self.general_params.shuffle_hop
            print("-----------------------")

    def get_audio_len(self, el, label):
        metainf = list(filter(lambda x: x.label == label, el))
        if not metainf:
            return 0

        duration = 0
        for m in metainf:
            start, end = m.from_to
            duration += end - start
        return duration

    def get_augsequence(self, augs):
        return AugSequence(
            [
                AUGMENTATIONS[aug](
                    **self.general_params.augmentations_config[aug], device=self.device
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
        train_dir = os.path.join(self.features_dir, self.general_params.train_folder)
        val_dir = os.path.join(self.features_dir, self.general_params.val_folder)
        test_dir = os.path.join(self.features_dir, self.general_params.test_folder)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        files = map(
            lambda x: os.path.join(self.features_dir, x), os.listdir(self.features_dir)
        )
        files = list(filter(os.path.isfile, files))

        train_ratio = int(len(files) * self.general_params.train_val_split[0] / 100)
        val_ratio = (
            int(len(files) * self.general_params.train_val_split[1] / 100) + train_ratio
        )
        test_ratio = len(files)
        self.dir_map = {
            train_dir: (0, train_ratio),
            val_dir: (train_ratio, val_ratio),
            test_dir: (val_ratio, test_ratio),
        }

        np.random.shuffle(files)
        [shutil.move(el, self.is_between(idx)) for idx, el in enumerate(files)]

    def audio_chunks_amount(self, length):
        return (length - self.general_params.mel_size // 1000) // (
            self.general_params.stride / 1000
        ) + 1

    def extract_features(self):
        self.general_params = munchify(self.config[GENERAL_PARAMS])
        self.device = "cuda" if self.general_params.use_gpu else "cpu"

        data_keys = list(self.config.keys())
        data_keys.remove(GENERAL_PARAMS)
        if not len(data_keys):
            raise Exception("No data points in config")

        dataset = {}
        for data_p in data_keys:
            data_info = munchify(self.config[data_p])
            if not os.path.exists(data_info.path):
                raise Exception(f"{data_info.path} directory doesn't exist")

            if not len(list(filter(len, data_info.tokens))):
                raise Exception(f"No tokens specified for {data_p}")

            if not os.path.exists(data_info.path) or not len(
                os.listdir(data_info.path)
            ):
                raise Exception(f"Data directory does not exits or empty for {data_p}")
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

            if self.general_params.use_equalization:
                [el.append("equalize_amplitude") for el in data_info.augmentations]

            augm_dict = {
                k: v for k, v in zip(data_info.tokens, data_info.augmentations)
            }

            if (
                not data_info.strict_token
                and data_info.label_file
                and data_info.audio_folder
            ):
                if (
                    not data_info.label_file in dir_content
                    or not data_info.audio_folder in dir_content
                ):
                    raise Exception(f"No label file or audio directory for {data_p}")

                try:
                    t_dataset = json.load(
                        open(join(data_info.path, data_info.label_file), "r")
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

                    t_dataset = {
                        os.path.join(data_info.path, data_info.audio_folder, k): list(
                            map(lambda x: x + [augm_dict[x[0]]], v)
                        )
                        for k, v in t_dataset.items()
                    }

                    dataset.update(t_dataset)
                except Exception as e:
                    raise e

            elif data_info.strict_token:
                if len(data_info.tokens) > 1:
                    raise Exception("Only one token for not labeled data is available")

                t_dataset = {}
                for audio in os.listdir(data_info.path):
                    audio_path = os.path.join(data_info.path, audio)

                    audio = sf.SoundFile(audio_path)
                    t_dataset.update(
                        {
                            audio_path: [
                                [
                                    data_info.strict_token,
                                    [0.0, np.round(len(audio) / audio.samplerate, 1)],
                                    augm_dict[data_info.strict_token],
                                ]
                            ]
                        }
                    )

                dataset.update(t_dataset)

            else:
                raise Exception("Config error")

        merging_dict = None
        if hasattr(self.general_params, "labels_merging"):
            merging_dict = dict(self.general_params.labels_merging)
            self.general_params.training_tokens = list(set(merging_dict.values())) + [
                el
                for el in self.general_params.training_tokens
                if el not in merging_dict
            ]

        if self.general_params.include_silence:
            positive_chunks = 0
            for params in dataset.values():
                for p in params:
                    if p[0] == self.general_params.training_tokens[-1]:
                        positive_chunks += self.audio_chunks_amount(p[1][1] - p[1][0])

            dataset.update(
                {
                    f"__silence__{str(i)}": [
                        [
                            "__silence__",
                            [0, self.general_params.stride / 1000],
                            ["gaussian_noise", "background_noise"],
                        ]
                    ]
                    for i in range(
                        int(self.general_params.include_silence * positive_chunks)
                    )
                }
            )
            self.general_params.training_tokens.insert(0, "__silence__")

        self.training_labels = {
            v: idx for idx, v in enumerate(self.general_params.training_tokens)
        }

        dataset = {
            k: list(
                map(
                    lambda x: munchify(
                        {
                            "label": merging_dict.get(x[0], x[0])
                            if merging_dict
                            else x[0],
                            "from_to": x[1],
                            "augs": x[2],
                        }
                    ),
                    v,
                )
            )
            for k, v in dataset.items()
        }

        dataset = [{item[0]: item[1]} for item in dataset.items()]
        np.random.shuffle(dataset)

        self.features_dir = self.general_params.output_path

        os.makedirs(self.features_dir, exist_ok=False)

        self.samples_per_square = self.ms2samples(
            self.general_params.mel_size, self.general_params.sample_rate
        )
        self.win_len = self.ms2samples(
            self.general_params.win_len, self.general_params.sample_rate
        )
        self.n_fft = self.general_params.n_fft
        self.hop_len = self.ms2samples(
            self.general_params.hop_len, self.general_params.sample_rate
        )

        self.stride = self.ms2samples(
            self.general_params.stride, self.general_params.sample_rate
        )

        self.feat_buff = []
        self.f_c = 0

        self.transform = FEATURE_TYPES[self.general_params.feature_type](
            sample_rate=self.general_params.sample_rate,
            win_length=self.win_len,
            n_fft=self.n_fft,
            hop_length=self.hop_len,
            n_mels=self.general_params.n_mels,
        )

        self.threads = (
            self.general_params.compute_threads
            if not self.general_params.use_gpu
            else 1
        )
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
            all(0 < el < 100 for el in self.general_params.train_val_split)
            and sum(self.general_params.train_val_split) < 100
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
