import argparse
from datetime import datetime, timedelta, time as date_time
import os
import time
from collections import defaultdict
import wave

import torch
import pyaudio
import numpy as np
import yaml
from munch import munchify
import pandas as pd


class StreamPrediction:
    def __init__(self, cfg_path):
        with open(cfg_path, encoding="utf-8") as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        self.cfg = munchify(config)
        self.chunk_samples = int(self.cfg.sample_rate * self.cfg.chunk_size / 1000)
        self.model = torch.jit.load(self.cfg.ckpt_path)

        self.logger = defaultdict(list)

    def start_stream(self):
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.cfg.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_samples,
        )

        os.makedirs(self.cfg.recorded_audio_path, exist_ok=True)
        os.makedirs(self.cfg.label_path, exist_ok=True)

        record_time = datetime.combine(datetime.now(), date_time.min)

        curr_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        record_wav = wave.open(
            os.path.join(self.cfg.recorded_audio_path, f"mic_record_{curr_time}.wav"),
            "w",
        )
        record_wav.setnchannels(1)
        record_wav.setsampwidth(2)
        record_wav.setframerate(self.cfg.sample_rate)

        label_file_name = os.path.join(
            self.cfg.label_path,
            f"mic_label_{curr_time}.xlsx",
        )

        stream.start_stream()

        try:
            while True:
                data = np.frombuffer(stream.read(self.chunk_samples), dtype="float32")
                data = np.expand_dims(data, 0)
                data = np.expand_dims(data, 0)

                with torch.no_grad():
                    pred = self.model(torch.tensor(data))

                record_wav.writeframes(
                    (np.squeeze(data, axis=0) * (2**15 - 1)).astype("<h")
                )
                record_time += timedelta(seconds=1)

                self.logger["Real Time"].append(str(datetime.now()))
                self.logger["Audio Time"].append(str(record_time))
                [
                    self.logger[f"Class {idx}"].append(el.item())
                    for idx, el in enumerate(torch.softmax(pred.squeeze(), -1))
                ]

                log_str = f"{datetime.now().strftime('%H:%M:%S:%f')} | {torch.softmax(pred, -1)};"

                print(
                    log_str,
                    sep="",
                    end="\n",
                    flush=True,
                )

                time.sleep(self.cfg.chunk_size / 1000)

        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()

            d = pd.DataFrame(self.logger)
            d.to_excel(label_file_name, index=False)

            record_wav.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config-path", type=str, help="Path to conf file", required=True
    )

    args = parser.parse_args()

    audio_stream = StreamPrediction(args.config_path)
    audio_stream.start_stream()
