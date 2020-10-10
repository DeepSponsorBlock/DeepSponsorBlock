import pathlib
import shutil
import subprocess

import pafy

INDICATOR_FILENAME = "completed"
DEFAULT_RESOLUTION = (256, 144)
DEFAULT_EXTENSION = "mp4"

class Video:
    def __init__(self, video_id, segments):
        self.video_id = video_id
        self.segments = segments
        self._pafy_obj = None

    @property
    def pafy_obj(self):
        if self._pafy_obj is None:
            self._pafy_obj = pafy.new(self.video_id)
        return self._pafy_obj

    @property
    def video_stream_url(self):
        video_streams = self.pafy_obj.videostreams
        chosen_stream = next(vs for vs in video_streams
                             if vs.extension == DEFAULT_EXTENSION
                             and vs.dimensions == DEFAULT_RESOLUTION)
        return chosen_stream.url

    def get_video_directory(self, root_path: pathlib.Path) -> pathlib.Path:
        return root_path / self.video_id

    def get_indicator_file_path(self, root_path: pathlib.Path) -> pathlib.Path:
        return self.get_video_directory(root_path) / INDICATOR_FILENAME

    def download(self, root_path, fps):
        indicator_file = self.get_indicator_file_path(root_path)
        if indicator_file.exists():
            return True

        video_directory = self.get_video_directory(root_path)

        # Delete the existing files, it's a partial download.
        if video_directory.exists():
            shutil.rmtree(video_directory)

        # Start downloading from scratch.
        video_directory.mkdir(parents=True)
        # '-q:v', '2',
        ffmpeg_args = ['ffmpeg', '-i', self.video_stream_url, '-vf', 'fps=%.4f' % fps, str((video_directory / "%d.jpg").absolute())]
        subprocess.run(ffmpeg_args, check=True, capture_output=True)

        # Go through the files and label them.
        for file in video_directory.glob("*.jpg"):
            # Get the file's timestamp
            timestamp = int(file.stem) / fps

            # Check if the file is sponsored
            is_sponsored = any(
                start <= timestamp < end for start, end in self.segments)

            # Rename the file now.
            file.rename(file.with_name("%s-%d.jpg" % (file.stem, int(is_sponsored))))

        # Save the finished indicator file.
        indicator_file.touch()

        return True


# ffmpeg -ss 00:00:30 -i $(youtube-dl -f 22 --get-url "$youtube_url") -vframes 1 -q:v 2 out.jpg