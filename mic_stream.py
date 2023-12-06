import argparse
from datetime import datetime, timedelta, time as date_time
import os
from collections import defaultdict
import wave
from queue import Queue

import torch
from torchaudio.transforms import Resample
import sounddevice as sd
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
        self.queue = Queue()

        self.default_mic_sr = int(
            sd.query_devices(self.cfg.default_mic_name)["default_samplerate"]
        )
        self.transform = Resample(
            orig_freq=self.default_mic_sr,
            new_freq=self.cfg.sample_rate,
        )
    
    def callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)

        audio = torch.t(torch.tensor(indata))
        audio = self.transform(audio)
        
        self.queue.put(torch.unsqueeze(audio, 0))

    def start_stream(self):
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

        try:
            with sd.Stream(
                device=self.cfg.default_mic_name,
                blocksize=self.default_mic_sr,
                dtype="float32",
                channels=1,
                callback=self.callback,
            ):
                while True:
                    data = self.queue.get()

                    with torch.no_grad():
                        pred = self.model(data)

                    record_wav.writeframes(
                        (np.squeeze(data.numpy(), axis=0) * (2**15 - 1)).astype("<h")
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
                    
        except (KeyboardInterrupt, SystemExit):
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
