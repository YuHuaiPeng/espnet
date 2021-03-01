# Dysarthric Voice Conversion using Voice Transformer Network (VTN)

We use a one-to-many version (o2m) of VTN, and we use the x-vector to control the speaker identity.

## Usage 

1. Data preparation. It is recommended to make sure the data preparation step is correctly executed, then start the training.
```
./run.sh --stage -1 --stop_stage 4 --cmvn downloads/cospro_tts_pt/data/train/cmvn.ark --dysarthric_download_url https://drive.google.com/open?id=<id>
```
The data preparation includes 6 stages:
    - Stage -1: Resources download and preprocessing. Dowdload the datasets (dysarthric and TMSV), TTS pretrained model and PWG model. Please ask Wen-Chin for the link to the dysarthric dataset. Also, normalize the power of all data. (THe dysarthric patient's voice is too loud.)
    - Stage 0: Basic data preparation. Generate `wav.scp`, `text`, `utt2spk`, etc.
    - Stage 1: VAD, feature generation, data split and normalization. We use a 240/40/40 data split. Remember to use the TTS pretrained model's `cmvn` file to do normalization.
    - Stage 2: Json Data Preparation.
    - Stage 3: x-vector extraction.
    - Stage 4: Merge Pair Json Data. We merge all data from the TMSV speakers for o2m training.

2. Training.
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 5 --stop_stage 5 --tag test
```

3. Decoding, synthesis and objective evaluation.
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 6 --tag test --cmvn downloads/cospro_tts_pt/data/train/cmvn.ark --model snapshot.ep.<ep>
```
3 stages are executed:
    - Stage 6: Decoding and synthesis. Remember to specify `--cmvn` for denormalization. If `--model` is not specified, the latest checkpoint will be automatically used. If you don't want to use PWG, specify `--voc GL` to use griffin-lim.
    - Stage 7: MCD evaluation.
    - Stage 8: ASR evaluation.

If you only want to do the evaluation, use the following:
```
CUDA_VISIBLE_DEVICES=0 ./run.sh --stage 7 --tag test --outdir exp/dysarthric_all_pytorch_test/outputs_snapshot.ep.<ep>_decode
```
