import logging
import os
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig

from openspeech.data.audio.data_loader import AudioDataLoader
from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.data.sampler import RandomSampler, SmartBatchingSampler
from openspeech.datasets import register_data_module

from openspeech.datasets.foreignkorean.character import generate_character_labels, generate_character_script
from openspeech.datasets.foreignkorean.preprocess import preprocess

logger = logging.getLogger(__name__)

@register_data_module("foreignkorean")
class LightningForeignKoreanDataModule(pl.LightningDataModule):
    FOREIGNKOREAN_TRAIN_NUM = 16870 #98.5%
    FOREIGNKOREAN_VALID_NUM = 775   #0.5%
    FOREIGNKOREAN_TEST_NUM = 1745   #1%

    def __init__(self, configs: DictConfig) -> None:
        super(LightningForeignKoreanDataModule, self).__init__()
        self.configs = configs
        self.dataset = dict()
        self.logger = logging.getLogger(__name__)
        self.encoding = "utf-8"

    def _generate_manifest_files(self, manifest_file_path: str) -> None:
        train_valid_audio_paths, train_valid_transcripts = preprocess(
            self.configs.dataset.dataset_path, self.configs.dataset.preprocess_mode
        )

        #test_audio_paths, test_transcripts = preprocess_test_data(
        #    self.configs.dataset.test_manifest_dir, self.configs.dataset.preprocess_mode
        #)

        audio_paths = train_valid_audio_paths #+ test_audio_paths
        transcripts = train_valid_transcripts #+ test_transcripts
        logger.info(f"audio_paths : {len(audio_paths)}")
        logger.info(f"transcripts : {len(transcripts)}")

        if self.configs.tokenizer.unit == "foreignkorean_character":
            generate_character_labels(transcripts, self.configs.tokenizer.vocab_path)
            generate_character_script(audio_paths, transcripts, manifest_file_path, self.configs.tokenizer.vocab_path)
        else:
            raise ValueError(f"Unsupported vocab : {self.configs.tokenizer.unit}")
        
    
    def _parse_manifest_file(self):
        audio_paths = list()
        transcripts = list()

        with open(self.configs.dataset.manifest_file_path, encoding=self.encoding) as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, korean_transcript, transcript = line.split("\t")
                transcript = transcript.replace("\n", "")

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts


    def prepare_data(self):
        print('prepare_data started..')
        #if not os.path.exists(self.configs.tokenizer.vocab_path):
        #    self._generate_vocab(self.configs.dataset.dataset_path)

        print(f"os.path.exists(self.configs.dataset.manifest_file_path) {os.path.exists(self.configs.dataset.manifest_file_path)}")
        print(f"os.path.exists(self.configs.dataset.dataset_path) {os.path.exists(self.configs.dataset.dataset_path)}")
        if not os.path.exists(self.configs.dataset.manifest_file_path):
            self.logger.error("Cannot find Manifest file")
            if not os.path.exists(self.configs.dataset.dataset_path):
                self.logger.error("Cannot find dataset path")
                raise FileNotFoundError
            self._generate_manifest_files(self.configs.dataset.manifest_file_path)

        print('prepare_data ended..')
    
    def setup(self, stage: Optional[str] = None) -> None:
        valid_end_idx = self.FOREIGNKOREAN_TRAIN_NUM + self.FOREIGNKOREAN_VALID_NUM
        audio_paths, transcripts = self._parse_manifest_file()
        audio_paths = {
            "train": audio_paths[: self.FOREIGNKOREAN_TRAIN_NUM],
            "valid": audio_paths[self.FOREIGNKOREAN_TRAIN_NUM : valid_end_idx],
            "test": audio_paths[valid_end_idx:],
        }
        transcripts = {
            "train": transcripts[: self.FOREIGNKOREAN_TRAIN_NUM],
            "valid": transcripts[self.FOREIGNKOREAN_TRAIN_NUM : valid_end_idx],
            "test": transcripts[valid_end_idx:],
        }

        for stage in audio_paths.keys():
            if stage == "test":
                dataset_path = self.configs.dataset.test_dataset_path
            else:
                dataset_path = self.configs.dataset.dataset_path

            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs,
                dataset_path=dataset_path,
                audio_paths=audio_paths[stage],
                transcripts=transcripts[stage],
                apply_spec_augment=self.configs.audio.apply_spec_augment if stage == "train" else False,
                del_silence=self.configs.audio.del_silence if stage == "train" else False,
            )
    
    def train_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        train_sampler = sampler(data_source=self.dataset["train"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["train"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        valid_sampler = sampler(self.dataset["valid"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["valid"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=valid_sampler,
        )

    def test_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        test_sampler = sampler(self.dataset["test"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["test"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=test_sampler,
        )