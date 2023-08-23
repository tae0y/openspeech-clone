CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python3 ./openspeech_cli/hydra_train.py \
dataset=foreignkorean \
dataset.dataset_path='/content/drive/MyDrive/openspeech-clone/openspeech/traindatasets' \
dataset.manifest_file_path='/content/drive/MyDrive/openspeech-clone/openspeech/traindatasets/manifest' \
dataset.test_dataset_path='/content/drive/MyDrive/openspeech-clone/openspeech/traindatasets' \
dataset.test_manifest_dir='/content/drive/MyDrive/openspeech-clone/openspeech/traindatasets' \
tokenizer=foreignkorean_character \
tokenizer.vocab_path='/content/drive/MyDrive/openspeech-clone/openspeech/traindatasets/vocab.csv' \
model=listen_attend_spell \
audio=melspectrogram \
lr_scheduler=warmup_reduce_lr_on_plateau \
trainer=tpu \
trainer.accelerator=auto \
trainer.max_epochs=1 \
trainer.sampler=random \
criterion=cross_entropy