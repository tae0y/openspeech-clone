import csv
from dataclasses import dataclass, field

from omegaconf import DictConfig

from openspeech.dataclass.configurations import TokenizerConfigs
from openspeech.tokenizers import register_tokenizer
from openspeech.tokenizers.tokenizer import Tokenizer


@dataclass
class ForeignKoreanCharacterTokenizerConfigs(TokenizerConfigs):
    unit: str = field(default="foreignkorean_character", metadata={"help": "Unit of vocabulary."})
    vocab_path: str = field(default="../../traindatasets/foreignkorean.csv", metadata={"help": "Path of vocabulary file."})


@register_tokenizer("kspon_character", dataclass=ForeignKoreanCharacterTokenizerConfigs)
class ForeignKoreanCharacterTokenizer(Tokenizer):
    def __init__(self, configs: DictConfig):
        super(ForeignKoreanCharacterTokenizer, self).__init__()
        self.vocab_dict, self.id_dict = self.load_vocab(
            vocab_path=configs.tokenizer.vocab_path,
            encoding=configs.tokenizer.encoding,
        )
        self.labels = self.vocab_dict.keys()
        self.sos_id = int(self.vocab_dict[configs.tokenizer.sos_token])
        self.eos_id = int(self.vocab_dict[configs.tokenizer.eos_token])
        self.pad_id = int(self.vocab_dict[configs.tokenizer.pad_token])
        self.blank_id = int(self.vocab_dict[configs.tokenizer.blank_token])
        self.vocab_path = configs.tokenizer.vocab_path

    def __len__(self):
        return len(self.labels)

    def decode(self, labels):
        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def encode(self, sentence):
        label = str()

        for ch in sentence:
            try:
                label += str(self.vocab_dict[ch]) + " "
            except KeyError:
                continue

        return label[:-1]

    def load_vocab(self, vocab_path, encoding="utf-8"):
        unit2id = dict()
        id2unit = dict()

        try:
            with open(vocab_path, "r", encoding=encoding) as f:
                labels = csv.reader(f, delimiter=",")
                next(labels)

                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(vocab_path))
