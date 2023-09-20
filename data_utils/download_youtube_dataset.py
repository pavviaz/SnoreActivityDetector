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


SNORE_TAG = "SNORE"
NO_SNORE_TAG = "NO_SNORE"
HOME_TAG = "HOME"
HUMAN_TAG = "HUMAN"

DATASET_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"

# https://github.com/audioset/ontology/blob/master/ontology.json
AUDIO_LABELS = {
    "/m/01d3sd": "Snore",
    "/m/02dgv": "Door",
    "/m/0642b4": "Cupboard open or close",
    "/m/0fqfqc": "Drawer open or close",
    "/m/04brg2": "Dishes, pots, and pans",
    "/m/023pjk": "Cutlery, silverware",
    "/m/07pn_8q": "Chopping (food)",
    "/m/0dxrf": "Frying (food)",
    "/m/0fx9l": "Microwave oven",
    "/m/02pjr4": "Blender",
    "/g/11b630rrvh": "Kettle whistle",
    "/m/02jz0l": "Water tap, faucet",
    "/m/0130jx": "Sink (filling or washing)",
    "/m/03dnzn": "Bathtub (filling or washing)",
    "/m/03wvsk": "Hair dryer",
    "/m/01jt3m": "Toilet flush",
    "/m/012xff": "Toothbrush",
    "/m/0d31p": "Vacuum cleaner",
    "/m/01s0vc": "Zipper (clothing)",
    "/m/0zmy2j9": "Velcro, hook and loop fastener",
    "/m/03v3yw": "Keys jangling",
    "/m/0242l": "Coin (dropping)",
    "/m/05mxj0q": "Packing tape, duct tape",
    "/m/01lsmm": "Scissors",
    "/m/081rb": "Writing",
    "/m/02g901": "Electric shaver, electric razor",
    "/m/05rj2": "Shuffling cards",
    "/m/0316dw": "Typing",
    "/m/0lyf6": "Breathing",
    "/m/07mzm6": "Wheeze",
    "/m/07s0dtb": "Gasp",
    "/m/07pyy8b": "Pant",
    "/m/07q0yl5": "Snort",
    "/m/01b_21": "Cough",
    "/m/0dl9sf8": "Throat clearing",
    "/m/01hsr_": "Sneeze",
    "/m/07ppn3j": "Sniff",
}

LABEL_MAPPING = {
    "Door": HOME_TAG,
    "Cupboard open or close": HOME_TAG,
    "Drawer open or close": HOME_TAG,
    "Dishes, pots, and pans": HOME_TAG,
    "Cutlery, silverware": HOME_TAG,
    "Chopping (food)": HOME_TAG,
    "Frying (food)": HOME_TAG,
    "Microwave oven": HOME_TAG,
    "Blender": HOME_TAG,
    "Kettle whistle": HOME_TAG,
    "Water tap, faucet": HOME_TAG,
    "Sink (filling or washing)": HOME_TAG,
    "Bathtub (filling or washing)": HOME_TAG,
    "Hair dryer": HOME_TAG,
    "Toilet flush": HOME_TAG,
    "Toothbrush": HOME_TAG,
    "Vacuum cleaner": HOME_TAG,
    "Zipper (clothing)": HOME_TAG,
    "Velcro, hook and loop fastener": HOME_TAG,
    "Keys jangling": HOME_TAG,
    "Coin (dropping)": HOME_TAG,
    "Packing tape, duct tape": HOME_TAG,
    "Scissors": HOME_TAG,
    "Writing": HOME_TAG,
    "Electric shaver, electric razor": HOME_TAG,
    "Shuffling cards": HOME_TAG,
    "Typing": HOME_TAG,
    "Breathing": HUMAN_TAG,
    "Wheeze": HUMAN_TAG,
    "Gasp": HUMAN_TAG,
    "Pant": HUMAN_TAG,
    "Snort": HUMAN_TAG,
    "Cough": HUMAN_TAG,
    "Throat clearing": HUMAN_TAG,
    "Sneeze": HUMAN_TAG,
    "Sniff": HUMAN_TAG,
    "Snore": SNORE_TAG,
}
YT_VIDEO_URL = "https://www.youtube.com/watch?v="


class DataDownloader:
    def __init__(self, config: dict):
        self.config = munchify(config)

    def get_intervals(self, from_timestamp, to_timestamp, duration, tag):
        from_timestamp, to_timestamp = float(from_timestamp), float(to_timestamp)
        if self.config.download_only_snore or (tag != SNORE_TAG):
            return [[tag, [0.0, to_timestamp - from_timestamp]]]

        res_label = [[SNORE_TAG, [from_timestamp, to_timestamp]]]
        if from_timestamp:
            res_label.append([NO_SNORE_TAG, [0.0, from_timestamp]])
        if to_timestamp != duration:
            res_label.append([NO_SNORE_TAG, [to_timestamp, duration]])

        return res_label

    def save_audio_from_vid(self, yt_vid_id, from_timestamp, to_timestamp, label, lock):
        try:
            label = LABEL_MAPPING[AUDIO_LABELS[label]]
            audio_path = None
            data_path = os.path.join(self.config.save_path, self.config.audios_dir)

            ydl_opts = {
                "quiet": True,
                "no-warnings": False,
                "ignore-errors": False,
                "no-overwrites": True,
                "outtmpl": os.path.join(data_path, f"{os.getpid()}.%(ext)s"),
                "format": "bestaudio/best",
                "keepvideo": False,
            }

            with Yt.YoutubeDL(ydl_opts) as ydl:
                metainf = ydl.extract_info(f"{YT_VIDEO_URL}{yt_vid_id}", download=True)

            audio_path = os.path.join(data_path, f"{os.getpid()}.{metainf['ext']}")

            ffmpeg_params = (
                ["ffmpeg"]
                + (
                    ["-ss", str(from_timestamp)]
                    if (self.config.download_only_snore or (label != SNORE_TAG))
                    else []
                )
                + ["-i", audio_path]
                + (
                    ["-to", str(to_timestamp - from_timestamp)]
                    if (self.config.download_only_snore or (label != SNORE_TAG))
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
                    os.path.join(self.config.save_path, self.config.tmp_labels), "a+"
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
        dict_temp = dct[start:end]
        for path in tqdm(dict_temp):
            self.save_audio_from_vid(*path, l)

    def main(self, dataset):
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
        def __get_label_line(line: str, choose_only_first_tag):
            if "#" in line:
                return "#"
            meta, tags = line.split(', "')
            re_tags = re.findall(r"[\/]\w[\/]\w+", tags)

            if not choose_only_first_tag:
                for tag in re_tags:
                    if tag in AUDIO_LABELS:
                        return f"{meta}, {tag}"
                return "#"

            return f"{meta}, {re_tags[0]}" if re_tags[0] in AUDIO_LABELS else "#"

        print(
            "Downloading and preprocessing data... Please wait, this can be time consuming..."
        )
        data = requests.get(DATASET_CSV_URL).text.split("\n")[:-1]
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
        with open(
            os.path.join(self.config.save_path, self.config.tmp_labels)
        ) as tmp_l, open(
            os.path.join(self.config.save_path, self.config.label_file), "w+"
        ) as l:
            json.dump(
                dict(ChainMap(*map(lambda x: eval(x.rstrip("\n")), tmp_l.readlines()))),
                l,
            )

    def download_and_save(self):
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

    with open(args.config_path) as c:
        config = yaml.load(c, Loader=yaml.FullLoader)

    downloader = DataDownloader(config)
    downloader.download_and_save()
