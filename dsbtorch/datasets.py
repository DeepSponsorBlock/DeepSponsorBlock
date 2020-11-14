import bisect
from collections import namedtuple
import math
import pathlib
from typing import List, Union, Any

import accimage
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm


ScannedDataset = namedtuple(
    "ScannedDataset",
    ["root_dir", "window_size", "n_indices", "cumulative_indices",
     "cumulative_dirs", "skip_every_n_pos", "skip_every_n_neg"])

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path: str) -> Any:
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def _get_file_label(f: pathlib.Path) -> int:
    """Gets the image label from the file path."""
    return int(f.stem.split("-")[1])

def scan_dataset(root_path: Union[pathlib.Path, str],
                 window_size: int,
                 skip_every_n_pos: int = 1,
                 skip_every_n_neg: int = 1) -> ScannedDataset:
    """
    Scans a directory for a DeepSponsorBlock dataset, formatted in the
    root_path/{video_id}/{frame_index}-{label}.jpg format, assuming that the
    caller wants to iterate over the dataset using a sliding window of frames
    containing window_size frames.

    :param root_path: Path of the dataset root directory.
    :param window_size: Size of the sliding window to iterate over the dataset.
    :param skip_every_n_pos: The number of frames between every two positive
        labeled frames. 1 returns every frame.
    :param skip_every_n_neg: The number of frames between every two negative
        labeled frames. 1 returns every frame.
    :param window_size: Size of the sliding window to iterate over the dataset.
    :return: ScannedDataset object for use in VideoSlidingWindowDataset or
        IterableVideoSlidingWindowDataset
    """
    if isinstance(root_path, str):
        root_path = pathlib.Path(root_path)

    if not root_path.exists() or not root_path.is_dir() or window_size <= 0:
        raise ValueError("Invalid input to scan_dataset.")

    # Keep track of the cumulative index reached to at each directory found in
    # the root.
    cumulative_indices = []
    cumulative_dirs = []

    cumulative_index = 0
    for directory in tqdm(list(root_path.iterdir())):
        if directory.is_dir():
            positive_files = list(directory.glob("*-1.jpg"))[::skip_every_n_pos]
            negative_files = list(directory.glob("*-0.jpg"))[::skip_every_n_neg]
            files = sorted(positive_files + negative_files, key=lambda x: int(x.stem.split("-")[0]))
            n_idx = len(files) - window_size + 1

            cumulative_indices.append(cumulative_index)
            cumulative_dirs.append(files)
            cumulative_index += n_idx

    return ScannedDataset(
        root_path, window_size, cumulative_index, cumulative_indices,
        cumulative_dirs, skip_every_n_pos, skip_every_n_neg)

def get_paths(sd: ScannedDataset, index: int) -> List[pathlib.Path]:
    """
    Gets the file paths of the images corresponding to the frames that the
    sliding window would contain if starting at the given image index.

    :param sd: ScannedDataset object obtained from scan_dataset.
    :param index: Index the sliding window would start at.
    :return: List[pathlib.Path] containing sd.window_size elements each
        corresponding to the respective image in the sliding window.
    """
    # Apply binary search to find the last directory that starts before index.
    directory_index = bisect.bisect_right(sd.cumulative_indices, index) - 1
    directory = sd.cumulative_dirs[directory_index]
    directory_start_index = sd.cumulative_indices[directory_index]

    # Compute the per-directory index of the global indices.
    start_index_in_directory = index - directory_start_index
    end_index_in_directory = start_index_in_directory + sd.window_size

    # Return the files at the given in-directory indices.
    return directory[start_index_in_directory:end_index_in_directory]

class VideoSlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, scanned_dataset: ScannedDataset,
                 transform=transforms.ToTensor()):
        self.sd: ScannedDataset = scanned_dataset
        self.transform = transform

    def __len__(self):
        return self.sd.n_indices

    def __getitem__(self, index):
        paths = get_paths(self.sd, index)
        image_list = [self.transform(accimage_loader(str(f))) for f in paths]

        images = torch.stack(image_list)
        labels = torch.tensor([_get_file_label(f) for f in paths])
        return (images, labels)


class IterableVideoSlidingWindowDataset(torch.utils.data.IterableDataset):
    def __init__(self, scanned_dataset: ScannedDataset,
                 transform=transforms.ToTensor()):
        self.sd: ScannedDataset = scanned_dataset
        self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.sd.n_indices
        else:
            per_worker = int(
                math.ceil((self.sd.n_indices) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.sd.n_indices)

        # Start iterating at the beginning of the iteration range.
        i = iter_start
        last_images = None
        last_labels = None

        # Now proceed through the dataset, yielding windows one-by-one.
        while i < iter_end:
            # Check if we're starting new video
            directory_index = bisect.bisect_right(self.sd.cumulative_indices, i) - 1
            directory_start_idx = self.sd.cumulative_indices[directory_index]

            paths = get_paths(self.sd, i)

            if i == directory_start_idx or i == iter_start:
                # The entire sliding window needs to be computed from scratch
                # if we have no prior window or the current index corresponds to
                # the starting index of a new directory.
                image_list = [self.transform(accimage_loader(str(f))) for f in paths]

                last_images = torch.stack(image_list)
                last_labels = torch.tensor([_get_file_label(f) for f in paths])
            else:
                # If we're just sliding the window by one in the same directory
                # then we can just drop the first item and append the new frame
                # as the last item.
                new_path = paths[-1]
                new_image = self.transform(accimage_loader(str(new_path)))

                last_images = torch.cat(
                    [last_images[1:], [new_image]])
                new_label = _get_file_label(new_path)
                last_labels = torch.cat(
                    [last_labels[1:], [new_label]])

            # Yield and move to the next window.
            yield (last_images, last_labels)
            i += 1

class PreEmbeddedDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: pathlib.Path):
        files = sorted(root_path.iterdir())

        embedding_files = [f for f in files if f.suffixes[0] == ".emb"]
        label_files = [f for f in files if f.suffixes[0] == ".lbl"]

        videos = list(zip(embedding_files, label_files))

        # Validate that the files are correctly paired.
        for emb_file, lbl_file in videos:
            assert emb_file.stem.split('.')[0] == lbl_file.stem.split('.')[0]

        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        emb_file, lbl_file = self.videos[index]

        embeddings = torch.tensor(np.load(emb_file))
        labels = torch.tensor(np.load(lbl_file))

        return (embeddings, labels)