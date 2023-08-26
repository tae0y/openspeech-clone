import os, re, json, logging

from joblib import Parallel, cpu_count, delayed
from tqdm import tqdm

rx = re.compile('[^가-힣\s]')
logger = logging.getLogger(__name__)

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
        if not os.path.isdir(path):
            continue

        filelist = os.listdir(path)
        #print(f"filelist >> {filelist}")
        for filename in filelist:
            filepath = os.path.join(path, os.path.splitext(filename)[0], filename + '.json')
            #print(f"filepath >> {filepath}")
            if os.path.exists(filepath):
                label_paths.append(filepath)
                audio_paths.append(filepath.replace('label','audio').replace('.json','.wav'))
                print(f"{filepath}, {filepath.replace('label','audio').replace('.json','.wav')}")
    
    #do parallel
    logger.debug(f"label_paths num : {len(label_paths)}")
    with Parallel(n_jobs=cpu_count()-1) as parallel:
        new_sentence = parallel(delayed(read_preprocess_text_file)(p.strip()) for p in label_paths)
        transcripts.extend(new_sentence)
    logger.debug(f"transcripts num : {len(transcripts)}")

    return audio_paths, transcripts

#def preprocess_test_data(manifest_file_dir: str, mode="character"):
#    audio_paths = list()
#    transcripts = list()
#
#    for split in ("eval_clean.trn", "eval_other.trn"):
#        with open(os.path.join(manifest_file_dir, split), encoding="utf-8") as f:
#            for line in f.readlines():
#                audio_path, raw_transcript = line.split("\t")
#                transcript = sentence_filter(raw_transcript)
#
#                audio_paths.append(audio_path)
#                transcripts.append(transcript)
#
#    return audio_paths, transcripts