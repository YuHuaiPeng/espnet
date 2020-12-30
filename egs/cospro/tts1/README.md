# COSPRO TTS recipe

## Data

The dataset is in `22.100`, `/mnt/md2/datasets/COSPRO_dataset`. If you wish to run it somewhere else, please copy the corpus.

## Environment

- xlrd 
    - Excel processing library. Used in `local/pinyin2ipa`.
    - `pip install xlrd==1.2.0` (Latest version causes error. 20201229)
- opencc
    - Traditional Chinese to simplified Chinese. Used for objective evaluation.
    - `pip install opencc`

## Usage

1. Install the abovementioned environmenet.
2. Set `datadir` correctly.
3. `CUDA_VISIBLE_DEVICES=0 ./run.sh --train_config conf/train_pytorch_transformer.yaml`

## Notes

1. Currently, the transcription type is IPA.
2. Syllable error rate is based on pinyin without tone.
