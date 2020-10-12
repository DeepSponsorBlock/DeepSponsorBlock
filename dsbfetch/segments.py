from collections import OrderedDict
import csv
import numpy as np
from .video import Video

def merge_overlapping_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)

    intervals.clear()
    intervals.extend(tuple(x) for x in merged)

def load_segments(filename):
    videos = OrderedDict()  # video_id -> Video

    # Read the videos & segments from the CSV file.
    with open(filename, "r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            video_id = row[0]

            # Get the timestamps.
            start_timestamp = float(row[1])
            end_timestamp = float(row[2])

            if video_id not in videos:
                videos[video_id] = Video(video_id, [])

            videos[video_id].segments.append(
                np.array([start_timestamp, end_timestamp]))

    # Correct overlapping segments.
    for video in videos.values():
        merge_overlapping_intervals(video.segments)

    return list(videos.values())