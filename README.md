# README
[openspeech의 원래 README는 여기에서](./README_origin.md)
  
## local cpu
- environments
  - m1 mac
  - python 3.10
- configuration
```shell
# 가상환경 만들기
python3 -m venv .venv
. .venv/bin/activate

# 의존성 설치
pip install -e .

# 패키지 일부 변경
pip install --upgrade torchmetric==0.6.0
pip install librosa
#collections.abc 관련오류는 venv에서 패키지 들어가서 코드를 수정해버림
#importlib는 제대로 동작안해서 무조건 True 반환하게끔 수정해버림
#colab에서 다시해보니 pytorch 버전을 올리면 되는 문제였음.........
```
- `sh./run.sh`
```shell
HYDRA_FULL_ERROR=1 python3 ./openspeech_cli/hydra_train.py \
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
criterion=cross_entropy
```

## colab tpu
- tip: 학습시키기전에 코드에서 학습데이터를 여러번 호출해서 서버에 캐싱해둔다
- configuration
```python
# 가상환경 만들기
!pip install -q condacolab
import condacolab
condacolab.install()

!conda create -n konda python==3.10
!source activate konda

# 의존성 설치
%cd /content/drive/MyDrive/openspeech-clone/
!pip install -e .

# 패키지 일부 변경
!pip install --upgrade torchmetrics==0.6.0
!pip install pytorch-lightning==1.8 -U
!pip install librosa
```