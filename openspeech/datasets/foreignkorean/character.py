
import logging

import pandas as pd

logger = logging.getLogger()


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += str(char2id[ch]) + " "
        except KeyError:
            continue

    return target[:-1]


def generate_character_labels(transcripts, labels_dest):
    logger.info("create_char_labels started..")

    label_list = list()
    label_freq = list()

    for transcript in transcripts:
        for ch in transcript:
            if ch not in label_list:
                label_list.append(ch)
                label_freq.append(1)
            else:
                label_freq[label_list.index(ch)] += 1

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {"id": [0, 1, 2, 3], "char": ["<pad>", "<sos>", "<eos>", "<blank>"], "freq": [0, 0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label["id"].append(idx + 4)
        label["char"].append(ch)
        label["freq"].append(freq)

    #label["id"] = label["id"][:2000]
    #label["char"] = label["char"][:2000]
    #label["freq"] = label["freq"][:2000]

    label_df = pd.DataFrame(label)
    label_df.to_csv(labels_dest, encoding="utf-8", index=False)


def generate_character_script(audio_paths: list, transcripts: list, manifest_file_path: str, vocab_path: str):
    logger.info("create_script started..")
    char2id, id2char = load_label(vocab_path)

    with open(manifest_file_path, "w") as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            char_id_transcript = sentence_to_target(transcript, char2id)
            f.write(f"{audio_path.strip()}\t{transcript}\t{char_id_transcript}\n")
