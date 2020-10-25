import bisect
from collections import namedtuple
import math
import pathlib
from typing import List, Union

import numpy as np
import skimage.io as io
import torch
from tqdm import tqdm


ScannedDataset = namedtuple(
    "ScannedDataset",
    ["root_dir", "window_size", "n_indices", "cumulative_indices",
     "cumulative_dirs"])

def _read_image(f: pathlib.Path) -> np.ndarray:
    return io.imread(str(f))

def _get_file_label(f: pathlib.Path) -> int:
    """Gets the image label from the file path."""
    return int(f.stem.split("-")[1])

def scan_dataset(root_path: Union[pathlib.Path, str],
                 window_size: int) -> ScannedDataset:
    """
    Scans a directory for a DeepSponsorBlock dataset, formatted in the
    root_path/{video_id}/{frame_index}-{label}.jpg format, assuming that the
    caller wants to iterate over the dataset using a sliding window of frames
    containing window_size frames.

    :param root_path: Path of the dataset root directory.
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
            n_files = len(list(directory.glob("*.jpg")))
            n_idx = n_files - window_size + 1

            cumulative_indices.append(cumulative_index)
            cumulative_dirs.append(directory)
            cumulative_index += n_idx

    return ScannedDataset(
        root_path, window_size, cumulative_index, cumulative_indices,
        cumulative_dirs)

def _get_paths(sd: ScannedDataset, index: int) -> List[pathlib.Path]:
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

    # Get the images inside this directory.
    files = list(directory.glob("*.jpg"))
    files.sort(key=lambda f: int(f.stem.split("-")[0])) # Integer part before -

    # Compute the per-directory index of the global indices.
    start_index_in_directory = index - directory_start_index
    end_index_in_directory = start_index_in_directory + sd.window_size

    # Return the files at the given in-directory indices.
    return files[start_index_in_directory:end_index_in_directory]

class VideoSlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, scanned_dataset: ScannedDataset, transform=None):
        self.sd: ScannedDataset = scanned_dataset
        self.transform = transform

    def __len__(self):
        return self.sd.n_indices

    def __getitem__(self, index):
        paths = _get_paths(self.sd, index)
        image_list = [_read_image(f) for f in paths]

        if self.transform:
            image_list = [self.transform(x) for x in image_list]

        images = torch.stack(image_list)
        labels = torch.tensor([_get_file_label(f) for f in paths])
        return (images, labels)


class IterableVideoSlidingWindowDataset(torch.utils.data.IterableDataset):
    def __init__(self, scanned_dataset: ScannedDataset, transform=None):
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

            paths = _get_paths(self.sd, i)

            if i == directory_start_idx or i == iter_start:
                # The entire sliding window needs to be computed from scratch
                # if we have no prior window or the current index corresponds to
                # the starting index of a new directory.
                image_list = [_read_image(f) for f in paths]

                if self.transform:
                    image_list = [self.transform(x) for x in image_list]

                last_images = torch.stack(image_list)
                last_labels = torch.tensor([_get_file_label(f) for f in paths])
            else:
                # If we're just sliding the window by one in the same directory
                # then we can just drop the first item and append the new frame
                # as the last item.
                new_path = paths[-1]
                new_image = _read_image(new_path)

                if self.transform:
                    new_image = self.transform(new_image)

                last_images = torch.cat(
                    [last_images[1:], [new_image]])
                new_label = _get_file_label(new_path)
                last_labels = torch.cat(
                    [last_labels[1:], [new_label]])

            # Yield and move to the next window.
            yield (last_images, last_labels)
            i += 1
