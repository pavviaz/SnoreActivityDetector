import multiprocessing
import subprocess
from tqdm import tqdm
import os
from glob import glob
import itertools


DATA_PATH = "E:/"
TARGET_PATH = "G:/"
SR = "16000"
CODEC = "pcm_s16le"
THREADS_AMOUNT = 1


def compute_threads_work(length, download_threads):
    """Yields split data for threads
    Args:
       length: number of captions in dataset
       download_threads: number of downloading threads
    Returns:
        (from, to) tuple
    """
    div, mod = divmod(length, download_threads)
    for _ in range(download_threads):
        yield (t := length - (div + bool(mod)), length)
        length = t
        mod -= 1 if mod else 0


def save_audio(video_path, out_path):
    # try:
    # os.makedirs(out_path, exist_ok=True)
    
    tmp_path = os.path.normpath(video_path)
    tmp_path = tmp_path.split(os.sep)
    
    subprocess.run(["ffmpeg", "-i", video_path, "-acodec", CODEC, "-vn", "-ar", SR, "-ac", "1", f"{os.path.join(out_path, tmp_path[-1].split('.')[0])}.wav"], 
                #    timeout=5, 
                capture_output=True)
            
    # except Exception as e:
    #     with open("errors.txt", "a") as err:
    #         err.write(f"ERROR: {e} with audio {video_path}\n")


def preprocess_and_save(dct, start, end):
    dict_temp = dct[start:end]
    for path in tqdm(dict_temp):
        save_audio(path)
        

def main(dataset):
    threads = [multiprocessing.Process(target=preprocess_and_save, args=(dataset, arg[0], arg[1])) for arg in
            compute_threads_work(len(dataset), THREADS_AMOUNT)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    folders = [glob(f"{os.path.join(DATA_PATH, el)}/**/*", recursive=True) for el in list(filter(lambda x: x.replace("_", "").isdigit(), os.listdir(DATA_PATH)))]
    folders = list(itertools.chain(*folders)) 
    folders = list(filter(os.path.isfile, folders))
    main(folders)