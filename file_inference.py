import json
import os
import time
from collections import defaultdict
import argparse

import numpy as np
import torch
from pydub import AudioSegment
import yaml
from munch import munchify

from data_processing import dataset_from_audio


class SnoreInference:
    def __init__(
        self,
        cfg_path: str,
    ):
        """
        Initializes the SnoreInference class
        object with the provided config.

        Args:
            cfg_path (str): Path to inference config.

        Returns:
            None. The method initializes the class object.
        """
        with open(cfg_path, encoding="utf-8") as c:
            config = yaml.load(c, Loader=yaml.FullLoader)

        self.cfg = munchify(config)
        self.model = torch.jit.load(self.cfg.ckpt_path)

    def cut_empty_audio(self, audio_path, output_path, predicted_VA):
        """
        Generates a clean audio file by removing empty segments
        from the original audio file based on the predicted snore activity.

        Args:
            audio_path (str): The path to the original audio file
            output_path (str): The path to save the generated clean audio file
            predicted_VA (dict): A dictionary containing the
            predicted snore activity segments

        Returns:
            None. The method generates a clean audio file based
            on the predicted snore activity segments
            and saves it to the specified output path.
        """
        original_audio = AudioSegment.from_file(audio_path, format="wav")
        original_audio = original_audio.set_frame_rate(self.cfg.sample_rate)

        clean_audio = AudioSegment.empty()
        stride_segments = defaultdict(list)

        print(f"Generating clear audio to {output_path}...")

        [
            [
                stride_segments[
                    (stride_segment - self.cfg.stride), stride_segment
                ].append(predicted_VA[p])
                for p in predicted_VA.keys()
                if p[0] < stride_segment <= p[-1]
            ]
            for stride_segment in range(
                self.cfg.stride,
                (1 + len(predicted_VA)) * self.cfg.stride,
                self.cfg.stride,
            )
        ]
        stride_segments = {
            k: round(np.mean(np.array(v).transpose(), axis=-1))
            for k, v in stride_segments.items()
        }

        for k, v in stride_segments.items():
            if v == 1:
                clean_audio += original_audio[k[0] : k[1]]

        clean_audio.export(f"{output_path}.wav", format="wav")
        print("Audio has been successfully generated")

    def labeling_func(self, x):
        return 1 if x[-1] > self.cfg.threshold else 0

    def predict(self, audio: str, output_file_name: str = None):
        """
        Predicts snore activity labels for
        an audio file and saves the labels in a JSON file.

        Args:
            audio (str): The path to the audio file for prediction.
            output_file_name (str, optional): The name of the output
            JSON file to save the labels. If not provided, a default
            name will be generated based on the audio
            file name, stride, and threshold.

        Returns:
            None
        """
        os.makedirs(self.cfg.label_path, exist_ok=True)
        audio_name = os.path.splitext(os.path.basename(audio))[0]
        output_file_name = os.path.join(
            self.cfg.label_path,
            f"{audio_name}_output_{self.cfg.stride}_stride_{self.cfg.threshold}_thr.json"
            if not output_file_name
            else output_file_name,
        )

        print(f"Processing {audio} with following config:\n{self.cfg}")

        labels = {}
        audio_dataset, segm_cnt = dataset_from_audio(
            audio,
            self.cfg.sample_rate,
            self.cfg.chunk_size,
            stride=self.cfg.stride,
            batch_size=self.cfg.batch_size,
        )
        print(f"{segm_cnt} segments were successfully extracted from audio")

        print(f"Starting inference...")
        t = time.time()
        for batch, (x_batch_val, y) in enumerate(audio_dataset):
            print(f"Processing batch {batch + 1} from {len(audio_dataset)}...")

            with torch.no_grad():
                output = torch.softmax(self.model(x_batch_val), dim=-1)
            labels.update(
                {
                    (
                        k * self.cfg.stride,
                        k * self.cfg.stride + self.cfg.chunk_size,
                    ): v
                    for k, v in zip(
                        y.cpu().numpy().tolist(), output.cpu().numpy().tolist()
                    )
                }
            )
        torch.cuda.empty_cache()
        print(f"Inference completed. Total time = {time.time() - t} seconds")

        if self.cfg.mean_overlaps:
            mean_segments = [
                [labels[p] for p in labels.keys() if p[0] < stride_segment <= p[-1]]
                for stride_segment in range(
                    self.cfg.stride,
                    (1 + len(labels)) * self.cfg.stride,
                    self.cfg.stride,
                )
            ]
            mean_segments = [
                np.mean(np.array(segm).transpose(), axis=-1).tolist()
                for segm in mean_segments
            ]

        labels = {
            k: [v, self.labeling_func(v)]
            if self.cfg.include_probs
            else self.labeling_func(v)
            for k, v in (
                labels.items()
                if not self.cfg.mean_overlaps
                else zip(labels.keys(), mean_segments)
            )
        }

        with open(output_file_name, "w+") as f:
            json.dump({str(k): v for k, v in labels.items()}, f, indent=4)

        if self.cfg.generate_clear_audio:
            os.makedirs(self.cfg.clear_audio_path, exist_ok=True)
            self.cut_empty_audio(
                audio,
                os.path.join(
                    self.cfg.clear_audio_path,
                    f"clear_{audio_name}_{self.cfg.generate_clear_audio}",
                ),
                labels
                if not self.cfg.include_probs
                else {k: v[-1] for k, v in labels.items()},
            )

        print("JSON labels file has been successfully created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config-path", type=str, help="Path to conf file", required=True
    )
    parser.add_argument(
        "-a",
        "--audio-path",
        type=str,
        help="Path to audio file to be labeled",
        required=True,
    )

    args = parser.parse_args()

    inf = SnoreInference(args.config_path)
    inf.predict(args.audio_path)
