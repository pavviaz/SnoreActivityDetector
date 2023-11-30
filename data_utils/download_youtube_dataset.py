import argparse
import multiprocessing
import os
import re
import subprocess
import requests
from collections import ChainMap
import json
import random

import yt_dlp as Yt
from tqdm import tqdm
import yaml
from munch import munchify

from multiprocess_funcs import compute_threads_work
import labels_config


TMP_LABELS_FILE = "tmp_labels.txt"


class DataDownloader:
    def __init__(self, config_path: str):
        with open(config_path, encoding='utf-8') as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        self.config = munchify(config)

    def get_intervals(
        self, from_timestamp: float, to_timestamp: float, duration: float, tag: str
    ):
        """
        Generate intervals based on the input timestamps and duration.

        Args:
            from_timestamp (float): The starting timestamp of the interval.
            to_timestamp (float): The ending timestamp of the interval.
            duration (float): The total duration of the audio.
            tag (str): The tag associated with the interval.

        Returns:
            list: A list of intervals with corresponding tags.
            Each interval is represented as a list with the tag
            as the first element and a list of two floats representing
            the start and end timestamps of the interval.
        """
        from_timestamp, to_timestamp = float(from_timestamp), float(to_timestamp)
        if self.config.download_only_snore or (tag != labels_config.SNORE_TAG):
            return [[tag, [0.0, to_timestamp - from_timestamp]]]

        res_label = [[labels_config.SNORE_TAG, [from_timestamp, to_timestamp]]]
        if from_timestamp:
            res_label.append([labels_config.NO_SNORE_TAG, [0.0, from_timestamp]])
        if to_timestamp != duration:
            res_label.append([labels_config.NO_SNORE_TAG, [to_timestamp, duration]])

        return res_label

    def save_audio_from_vid(
        self,
        yt_vid_id: str,
        from_timestamp: float,
        to_timestamp: float,
        label: str,
        lock: multiprocessing.Lock,
    ):
        """
        Downloads and saves audio from a YouTube video
        based on the specified timestamps and label.

        Args:
            yt_vid_id (str): The YouTube video ID.
            from_timestamp (float): The start timestamp of
            the audio segment to be saved.
            to_timestamp (float): The end timestamp of the
            audio segment to be saved.
            label (str): The label/tag associated with the audio segment.
            lock (multiprocessing.Lock): A lock object for thread synchronization.

        Returns:
            None. The method saves the trimmed audio segment as a
            WAV file and appends the label information to a temporary label file.
        """
        try:
            label = labels_config.LABEL_MAPPING[labels_config.AUDIO_LABELS[label]]
            audio_path = None
            data_path = os.path.join(self.config.save_path, self.config.audios_dir)

            ydl_opts = {
                "quiet": True,
                "no-warnings": False,
                "ignore-errors": False,
                "no-overwrites": True,
                "outtmpl": os.path.join(data_path, f"{os.getpid()}.%(ext)s"),
                "format": "best",
                "keepvideo": False,
            }

            with Yt.YoutubeDL(ydl_opts) as ydl:
                metainf = ydl.extract_info(
                    f"{labels_config.YT_VIDEO_URL}{yt_vid_id}", download=True
                )

            audio_path = os.path.join(data_path, f"{os.getpid()}.{metainf['ext']}")

            ffmpeg_params = (
                ["ffmpeg"]
                + (
                    ["-ss", str(from_timestamp)]
                    if (
                        self.config.download_only_snore
                        or (label != labels_config.SNORE_TAG)
                    )
                    else []
                )
                + ["-i", audio_path]
                + (
                    ["-to", str(to_timestamp - from_timestamp)]
                    if (
                        self.config.download_only_snore
                        or (label != labels_config.SNORE_TAG)
                    )
                    else []
                )
                + ["-acodec", self.config.codec]
                + ["-ar", str(self.config.sample_rate)]
                + ["-ac", "1"]
                + [f"{os.path.join(data_path, yt_vid_id)}.wav"]
            )

            subprocess.run(ffmpeg_params, timeout=5, capture_output=True)

            with lock:
                label_stream = open(
                    os.path.join(self.config.save_path, TMP_LABELS_FILE), "a+"
                )
                label_stream.write(
                    str(
                        {
                            f"{yt_vid_id}.wav": self.get_intervals(
                                from_timestamp,
                                to_timestamp,
                                float(metainf["duration"]),
                                label,
                            )
                        }
                    )
                    + "\n"
                )
                label_stream.close()

        except Exception as exc:
            with lock:
                if "Удаленный" in str(exc):
                    exc = (
                        "Удаленный хост принудительно разорвал существующее подключение"
                    )

                print(f"Error {exc} with {yt_vid_id} video\n")
                error_stream = open(
                    os.path.join(self.config.save_path, self.config.error_log), "a+"
                )
                error_stream.write(f"Error {exc} with {yt_vid_id} video\n")
                error_stream.close()

        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

    def preprocess_and_save(self, dct, l, start, end):
        """
        Preprocesses a subset of the given
        dictionary and saves the corresponding audio segments.

        Args:
            dct (dict): A dictionary containing audio segment data.
            l (str): A label associated with the audio segments.
            start (int): The starting index of the subset of data to be processed.
            end (int): The ending index of the subset of data to be processed.

        Returns:
            None. The method processes a subset of
            the data and saves the corresponding audio segments.
        """
        dict_temp = dct[start:end]
        for path in tqdm(dict_temp):
            self.save_audio_from_vid(*path, l)

    def main(self, dataset: list):
        """
        Downloads and saves audio segments
        from YouTube videos based on the provided dataset.

        Args:
            dataset (list): A list of audio segments to be downloaded and saved.
            Each audio segment is represented as a list with the following elements:
            YouTube video ID, start timestamp, end timestamp, and label.

        Returns:
            None. The method downloads and saves the audio
            segments as WAV files and appends the label
            information to a temporary label file.
        """
        lock = multiprocessing.Lock()

        threads = [
            multiprocessing.Process(
                target=self.preprocess_and_save, args=(dataset, lock, arg[0], arg[1])
            )
            for arg in compute_threads_work(
                len(dataset), self.config.download_threads_amount
            )
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def get_dataset_csv(self, choose_only_first_tag=False):
        """
        Retrieves and processes a dataset in CSV format from a specified URL.

        Args:
            choose_only_first_tag (bool, optional): A boolean flag indicating whether
            to choose only the first tag from the available
            tags in each line of the dataset. Default is False.

        Returns:
            list: A list of processed data, where each element
            represents an audio segment and contains the following information:
            YouTube video ID, start timestamp, end timestamp, and label.
        """

        def __get_label_line(line: str, choose_only_first_tag):
            if "#" in line:
                return "#"
            meta, tags = line.split(', "')
            re_tags = re.findall(r"[\/]\w[\/]\w+", tags)

            if not choose_only_first_tag:
                for tag in re_tags:
                    if tag in labels_config.AUDIO_LABELS:
                        return f"{meta}, {tag}"
                return "#"

            return (
                f"{meta}, {re_tags[0]}"
                if re_tags[0] in labels_config.AUDIO_LABELS
                else "#"
            )

        print(
            "Downloading and preprocessing data... Please wait, this can be time consuming..."
        )
        data = requests.get(labels_config.DATASET_CSV_URL).text.split("\n")[:-1]
        random.shuffle(data)

        data = map(lambda x: __get_label_line(x, choose_only_first_tag), data)
        data = filter(lambda x: not "#" in x, data)

        data = map(lambda x: x.replace(" ", "").split(","), data)
        data = map(
            lambda x: [x[0], int(x[1].split(".")[0]), int(x[2].split(".")[0]), x[3]],
            data,
        )
        return list(data)

    def assemble_label_file(self):
        """
        Assembles the label file by reading the temporary label file,
        processing its contents, and writing
        the final label file in JSON format.

        Args:
            self: The instance of the DataDownloader class.

        Returns:
            None. The method writes the final label file in JSON format.
        """
        with open(
            os.path.join(self.config.save_path, TMP_LABELS_FILE)
        ) as tmp_l, open(
            os.path.join(self.config.save_path, self.config.label_file), "w+"
        ) as l:
            json.dump(
                dict(ChainMap(*map(lambda x: eval(x.rstrip("\n")), tmp_l.readlines()))),
                l,
            )

    def download_and_save(self):
        """
        Downloads and saves audio segments
        from YouTube videos based on the dataset.

        Returns:
            None. The method creates the necessary directories,
            downloads and saves the audio segments, and writes the final label file.
        """
        os.makedirs(self.config.save_path, exist_ok=True)
        os.makedirs(
            os.path.join(self.config.save_path, self.config.audios_dir), exist_ok=True
        )

        data = self.get_dataset_csv(choose_only_first_tag=self.config.only_first)
        self.main(data)

        self.assemble_label_file()


# python3 download_youtube_dataset.py --config-path data_download_configs/download_of.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path", type=str, help="Path to conf file", required=True
    )

    args = parser.parse_args()

    downloader = DataDownloader(args.config_path)
    downloader.download_and_save()
