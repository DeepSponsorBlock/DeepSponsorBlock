import pathlib
import sys
from typing import Tuple

import torch
from tqdm import tqdm

import dsbtorch
from . import Video

YOUTUBE_LINK_FORMAT = "https://www.youtube.com/embed/%s?start=%d&end=%d"
DEVICE_NAME = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_NAME)
WEIGHTS_PATH = pathlib.Path(__file__).parent.parent.absolute() / "results"


def predict(
        video_id: str,
        root_path: pathlib.Path,
        batch_size: int = 1024,
        verbose: bool = False) -> Tuple[int, int]:
    # Download the video.
    if verbose:
        print("Downloading video.")
    video = Video(video_id)
    status, err = video.download(root_path, 1, False)
    if not status:
        raise ValueError("Could not download video: %s" % err)

    # Create the dataset object.
    if verbose:
        print("Parsing dataset.")
    sd = dsbtorch.scan_dataset(root_path, 1, 1, 1)
    dataset = dsbtorch.VideoSlidingWindowDataset(
        sd, dsbtorch.DEFAULT_TRANSFORM, get_labels=False)

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=6, pin_memory=True)

    # Load the ResCNN model
    if verbose:
        print("Loading ResCNN.")
    cnn_path = WEIGHTS_PATH / "rescnn.weights"
    cnn = dsbtorch.ResCNN(
        weights_path=cnn_path, map_location={'cuda:0': DEVICE_NAME}).to(DEVICE)
    cnn.eval()

    if verbose:
        print("Running ResCNN on video frames.")
    all_encodings = []
    with torch.set_grad_enabled(False):
        for images in tqdm(dataloader, file=sys.stdout):
            # Reshape it and put on device.
            images = images.squeeze(dim=1).to(DEVICE)

            encodings = cnn(images)
            all_encodings.append(encodings)

    # Prepare to feed the data to the Encoder/Decoder.
    all_encodings = torch.cat(all_encodings).to(DEVICE)

    # Load the Encoder/Decoder model.
    if verbose:
        print("Loading Encoder/Decoder.")
    rnn_path = WEIGHTS_PATH / "preprocessed_encoder_decoder.weights"
    rnn = dsbtorch.PreprocessedEncoderDecoder(
        in_features=cnn.fc.in_features, sigmoid=False,
        weights_path=rnn_path, map_location={'cuda:0': DEVICE_NAME}).to(DEVICE)
    rnn.eval()

    if verbose:
        print("Running Encoder/Decoder on preprocessed videos.")
    with torch.set_grad_enabled(False):
        start_labels, end_labels = rnn(all_encodings)

    start_seconds = torch.argmax(start_labels.view(-1))
    end_seconds = torch.argmax(start_labels.view(-1))

    if end_seconds < start_seconds:
        raise ValueError("Could not predict sponsored segment.")

    return int(start_seconds), int(end_seconds)


def get_youtube_link(video_id, start, end):
    return YOUTUBE_LINK_FORMAT % (video_id, start, end)
