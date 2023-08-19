import os, re, json

from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

rx = re.compile('[^가-힣\s]')

def sentence_filter(raw_sentence):
    return re.sub(rx, '', raw_sentence)

def read_preprocess_text_file(file_path):
    with open(file_path, "r", encoding="utf8") as json_file:
        json_data = json.load(json_file)
        readTxt = json_data["transcription"]["ReadingLabelText"] #대본리딩
        answerTxt = json_data["transcription"]["AnswerLabelText"] #질의응답
        transscript = readTxt if len(readTxt.strip())>0 else answerTxt
        return sentence_filter(transscript)
    
def preprocess(dataset_path, mode="character"):
    print("preprocess started..")
    
    #dataset_path : openspeech/openspeech/traindataset
    #subdir       : audio, label
    #subsubdir    : 1, 2, 3, 4, 5
    workdir = os.path.join(dataset_path, 'label')
    subdirs = os.listdir(workdir)
    label_paths = []
    audio_paths = list()
    transcripts = list()
    for dir in subdirs:
        path = os.path.join(workdir, dir)
        if not os.path.isdir(dir):
            continue

        #insde subsubdir : list.txt, ~~~.json, ~~~~.csv
        listpath = os.path.join(path, 'list.txt')
        with open(listpath, 'r') as f:
            for filename in f.readlines():
                if filename.endswith(".json"):
                    filepath = os.path.join(path, filename)
                    label_paths.append(filepath)
                    audio_paths.append(filepath.rplace('label','audio').replace('.json','.wav'))
    
    #do parallel
    with Parallel(n_jons=cpu_count()-1) as parallel:
        new_sentence = parallel(delayed(read_preprocess_text_file)(p) for p in label_paths)
        transcripts.extend(new_sentence)

    return audio_paths, transcripts

def preprocess_test_data(manifest_file_dir: str, mode="character"):
    audio_paths = list()
    transcripts = list()

    for split in ("eval_clean.trn", "eval_other.trn"):
        with open(os.path.join(manifest_file_dir, split), encoding="utf-8") as f:
            for line in f.readlines():
                audio_path, raw_transcript = line.split("\t")
                transcript = sentence_filter(raw_transcript)

                audio_paths.append(audio_path)
                transcripts.append(transcript)

    return audio_paths, transcripts