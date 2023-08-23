CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python3 ./openspeech_cli/hydra_train.py \
dataset=foreignkorean \
dataset.dataset_path='/Users/bachtaeyeong/PROJECTS/vietnam-audio/openspeech/openspeech/traindatasets' \
dataset.manifest_file_path='/Users/bachtaeyeong/PROJECTS/vietnam-audio/openspeech/openspeech/traindatasets/manifest' \
dataset.test_dataset_path='/Users/bachtaeyeong/PROJECTS/vietnam-audio/openspeech/openspeech/traindatasets' \
dataset.test_manifest_dir='/Users/bachtaeyeong/PROJECTS/vietnam-audio/openspeech/openspeech/traindatasets' \
tokenizer=foreignkorean_character \
tokenizer.vocab_path='/Users/bachtaeyeong/PROJECTS/vietnam-audio/openspeech/openspeech/traindatasets/vocab.csv' \
model=listen_attend_spell \
audio=melspectrogram \
lr_scheduler=warmup_reduce_lr_on_plateau \
trainer=cpu \
trainer.accelerator=auto \
trainer.max_epochs=1 \
trainer.sampler=random \
trainer.batch_size=2 \
criterion=cross_entropy