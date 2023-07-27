import multiprocessing
import os
import subprocess
from tqdm import tqdm
import yt_dlp as Yt
import requests
from utils.multiprocess_funcs import compute_threads_work
from collections import ChainMap
import json
import random
from itertools import islice, chain


DOWNLOAD_ONLY_SNORE = False
SNORE_TAG = "SNORE"
NO_SNORE_TAG = "NO_SNORE"
HOME_TAG = "HOME"

THREADS_AMOUNT = 1
DATASET_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
SNORE_LABEL = "/m/01d3sd"  # snoring (https://github.com/audioset/ontology/blob/master/ontology.json)
HOME_LABELS = ["/m/02dgv", "/m/0642b4", "/m/0fqfqc", "/m/04brg2", "/m/023pjk", "/m/07pn_8q", 
               "/m/0dxrf", "/m/0fx9l", "/m/02pjr4", "/g/11b630rrvh", "/m/02jz0l", "/m/0130jx", 
               "/m/03dnzn", "/m/03wvsk", "/m/01jt3m", "/m/012xff", "/m/0d31p", "/m/01s0vc", 
               "/m/0zmy2j9", "/m/03v3yw", "/m/0242l", "/m/05mxj0q", "/m/01lsmm", "/m/02g901", 
               "/m/05rj2", "/m/0316dw", "/m/081rb"]
YT_VIDEO_URL = "https://www.youtube.com/watch?v="

SR = "16000"
CODEC = "pcm_s16le"  # change to 24 bit

DATASET_DIR = "data_home"
AUDIOS_DIR = "audios"
TMP_LABELS = "tmp_labels.txt"
LABEL_FILE = "labels.json"
ERROR_LOG = "errors.txt"


lock = multiprocessing.Lock()


def get_intervals(from_timestamp, to_timestamp, duration, is_home):
    from_timestamp, to_timestamp = float(from_timestamp), float(to_timestamp)
    if DOWNLOAD_ONLY_SNORE or is_home:
        return [[HOME_TAG if is_home else SNORE_TAG, [0.0, to_timestamp - from_timestamp]]]
    
    res_label = [[SNORE_TAG, [from_timestamp, to_timestamp]]]
    if from_timestamp:
        res_label.append([NO_SNORE_TAG, [0.0, from_timestamp]])
    if to_timestamp != duration:
        res_label.append([NO_SNORE_TAG, [to_timestamp, duration]])
    
    return res_label
    

def save_audio_from_vid(yt_vid_id, from_timestamp, to_timestamp, label):
    try:
        is_home = label != SNORE_LABEL
        audio_path = None
        data_path = os.path.join(DATASET_DIR, AUDIOS_DIR)

        ydl_opts = {
            'quiet': True,
            'no-warnings': False,
            'ignore-errors': False,
            'no-overwrites': True,
            'outtmpl': os.path.join(data_path, f'{os.getpid()}.%(ext)s'),
            'format': 'bestaudio/best',
            'keepvideo': False
        }
        
        with Yt.YoutubeDL(ydl_opts) as ydl:
            metainf = ydl.extract_info(f"{YT_VIDEO_URL}{yt_vid_id}", download=True)
        
        audio_path = os.path.join(data_path, f"{os.getpid()}.{metainf['ext']}")
        
        ffmpeg_params = ["ffmpeg"] + \
                        (["-ss", str(from_timestamp)] if (DOWNLOAD_ONLY_SNORE or is_home) else []) + \
                        ["-i", audio_path] + \
                        (["-to", str(to_timestamp - from_timestamp)] if (DOWNLOAD_ONLY_SNORE or is_home) else []) + \
                        ["-acodec", CODEC] + \
                        ["-ar", SR] + \
                        ["-ac", "1"] + \
                        [f"{os.path.join(data_path, yt_vid_id)}.wav"]

        subprocess.run(ffmpeg_params, timeout=5, capture_output=True)

        lock.acquire()
        label_stream = open(os.path.join(DATASET_DIR, TMP_LABELS), "a+")
        label_stream.write(str({f"{yt_vid_id}.wav": get_intervals(from_timestamp, to_timestamp, float(metainf["duration"]), is_home)}) + "\n")
        label_stream.close()
        lock.release()
        
    except Exception as exc:
        lock.acquire()
        if "Удаленный" in str(exc):
            exc = "Удаленный хост принудительно разорвал существующее подключение"

        print(f"Error {exc} with {yt_vid_id} video\n")
        error_stream = open(os.path.join(DATASET_DIR, ERROR_LOG), "a+")
        error_stream.write(f"Error {exc} with {yt_vid_id} video\n")
        error_stream.close()
        lock.release()
    
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def preprocess_and_save(dct, start, end):
    dict_temp = dct[start:end]
    for path in tqdm(dict_temp):
        save_audio_from_vid(*path)
        

def main(dataset):
    threads = [multiprocessing.Process(target=preprocess_and_save, args=(dataset, arg[0], arg[1])) for arg in
            compute_threads_work(len(dataset), THREADS_AMOUNT)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def get_dataset_csv():
    def __get_label_line(line: str):
        meta, tags = line.split(', "')
        if len((t:=tags.split(","))) > 1:
            return f"{meta}, {t[0]}"
        return f'{meta}, {tags[:-1]}'

    print("Downloading and preprocessing data... Please wait, this can be time consuming...")
    data = requests.get(DATASET_CSV_URL).text.split("\n")
    random.shuffle(data)

    snore_data = list(filter(lambda x: SNORE_LABEL in x and not "#" in x, data))
    home_data = filter(lambda x: any(tag in x for tag in HOME_LABELS) and not "#" in x, data)
    data = chain(snore_data, islice(home_data, len(snore_data)))

    data = map(__get_label_line, data)
    data = map(lambda x: x.replace(" ", "").split(","), data)
    data = map(lambda x: [x[0], int(x[1].split(".")[0]), int(x[2].split(".")[0]), x[3]], data)
    return list(data)


def assemble_label_file():
    with open(os.path.join(DATASET_DIR, TMP_LABELS)) as tmp_l, open(os.path.join(DATASET_DIR, LABEL_FILE), "w+") as l:
        json.dump(dict(ChainMap(*map(lambda x: eval(x.rstrip("\n")), tmp_l.readlines()))), l)


if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, AUDIOS_DIR), exist_ok=True)

    data = get_dataset_csv()
    main(data)

    assemble_label_file()
