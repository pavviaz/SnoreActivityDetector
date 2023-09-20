import argparse
import multiprocessing
import subprocess
import os

from tqdm import tqdm
import yaml
from munch import munchify

from multiprocess_funcs import compute_threads_work


ERROR_LOG = "errors.txt"


class WavExtractor:
    def __init__(self, config_path):
        with open(config_path) as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        self.config = munchify(config)

    def save_audio(self, video_path, lock):
        try:
            tmp_path = os.path.normpath(video_path)
            tmp_path = tmp_path.split(os.sep)

            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-acodec",
                    self.config.codec,
                    "-vn",
                    "-ar",
                    str(self.config.sample_rate),
                    "-ac",
                    "1",
                    f"{os.path.join(self.config.save_path, tmp_path[-1].split('.')[0])}.wav",
                ],
                timeout=5,
                capture_output=True,
            )

        except Exception as e:
            with lock:
                err = open(os.path.join(self.config.save_path, ERROR_LOG), "a")
                err.write(f"ERROR: {e} with audio {video_path}\n")
                err.close()

    def preprocess_and_save(self, dct, l, start, end):
        dict_temp = dct[start:end]
        for path in tqdm(dict_temp):
            self.save_audio(path, l)

    def main(self, dataset):
        lock = multiprocessing.Lock()

        threads = [
            multiprocessing.Process(
                target=self.preprocess_and_save, args=(dataset, lock, arg[0], arg[1])
            )
            for arg in compute_threads_work(
                len(dataset), self.config.extracting_threads_amount
            )
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def extract_and_save(self):
        folders = [
            os.path.join(self.config.data_path, el)
            for el in os.listdir(self.config.data_path)
        ]
        folders = list(filter(os.path.isfile, folders))
        self.main(folders)


# python3 extract_wav_from_video.py --config-path wav_extraction_configs/extract_wav.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-path", type=str, help="Path to conf file", required=True
    )

    args = parser.parse_args()

    extractor = WavExtractor(args.config)
    extractor.extract_and_save()
