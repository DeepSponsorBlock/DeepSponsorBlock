# DeepSponsorBlock
Deep learning-based solution for identifying sponsored content segments on YouTube videos.

This repository contains our [models](dsbtorch/train/models.py), [training scripts](notebooks/), [pretrained weights](results/), [dataset downloader](dsbfetch/video.py) and [prediction script](dsbfetch/predict.py) (which also demonstrates the end-to-end flow).

Click [here](https://www.youtube.com/watch?v=gvpOPB_hhxo) for a video detailing the architecture and results.

This project was completed in Autumn 2020 for Stanford's CS 230 (Deep Learning) course by Nikhil Athreya, Cem Gokmen and Jennie Yang.

## Running predictions
Run the `dsbfetch predict` command as follows to run a prediction by video ID using our pretrained weights. Note that this will download the video using FFmpeg and run the ResNet and the decoder on it, which works well on CUDA. You can change the ResNet batch size using the `--batch-size` option if the current batch size causes memory issues.

```console
~$ python -m dsbfetch predict -v Qa0jZnrQrIA
Downloading video.
Parsing dataset.
100%|█████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 141.14it/s]
Loading ResCNN.
Running ResCNN on video frames.
100%|█████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.60s/it]
Loading Encoder/Decoder.
Running Encoder/Decoder on preprocessed videos.

Prediction successful. Segment link: https://www.youtube.com/embed/Qa0jZnrQrIA?start=45&end=66
```

## Fetching the dataset
The dataset fetcher is designed to work with Python 3.7.9 and requires `ffmpeg` to be installed.

Start by installing the requirements:
```
pip install -r requirements.txt
```

Then use the below command to see all options:
```
python -m dsbfetch
```

Example invocation with typical inputs:
```
python -m dsbfetch --limit-count=20 segments.csv downloads/
```

## Training the model
The training process takes a few steps due to the complicated architecture.

1) Index the dataset (and save the index) using the [PrescanDatasets](notebooks/PrescanDatasets.ipynb) notebook.
2) Train the ResNet (possibly on a subset of the dataset) using the [TrainResNet](notebooks/TrainResNet.ipynb) notebook.
3) Use the [ApplyResNet](notebooks/ApplyResNet.ipynb) notebook to pre-process the dataset by feeding all frames through the ResNet and storing the output encodings.
4) Train the Encoder/Decoder architecture using the pre-processed dataset using the [TrainRNN](notebooks/TrainRNN.ipynb) notebook.
5) Evaluate the ResNet as a baseline using the [EvalCNN](notebooks/EvalCNN.ipynb) notebook and the full Encoder/Decoder architecture using the [EvalRNN](notebooks/EvalRNN.ipynb).
6) You can now use the trained models through the prediction script as well.

## Licensing

Data used to train this model and `segments.csv` use the [SponsorBlock](https://sponsor.ajay.app/) dataset by [Ajay Ramachandran](https://ajay.app/) and are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
