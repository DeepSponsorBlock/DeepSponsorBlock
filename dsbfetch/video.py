import pathlib
import shutil

import ffmpeg
import pafy

INDICATOR_FILENAME = "completed"
STDERR_FILENAME = "stderr.log"
DEFAULT_RESOLUTION = (256, 144)
DEFAULT_EXTENSION = "mp4"

class Video:
    def __init__(self, video_id, segments):
        self.video_id = video_id
        self.segments = segments
        self._pafy_obj = None

    def __str__(self):
        return self.video_id

    def __repr__(self):
        return "Video %s" % self.video_id

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

    def is_already_downloaded(self, root_path: pathlib.Path) -> bool:
        return self.get_indicator_file_path(root_path).exists()

    def download(self, root_path, fps, log_ffmpeg):
        try:
            if self.is_already_downloaded(root_path):
                return True, None

            video_directory = self.get_video_directory(root_path)

            # Delete the existing files, it's a partial download.
            if video_directory.exists():
                shutil.rmtree(video_directory)

            # Do this first so that the lazy eval happens & if it causes any issues
            # we don't end up with a broken file.
            stream_url = self.video_stream_url

            # Start downloading from scratch.
            video_directory.mkdir(parents=True)
            out_filename = str((video_directory / "%d.jpg").absolute())
            _, stderr = (ffmpeg
              .input(stream_url)
              .filter('fps', fps='%.4f' % fps)
              .output(out_filename, **{'qscale:v': 2})
              .run(quiet=True))

            # Log errors.
            if log_ffmpeg:
                with open(video_directory / STDERR_FILENAME, 'w') as stderr_file:
                    stderr_file.write(str(stderr))

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
            self.get_indicator_file_path(root_path).touch()

            return True, None
        except BaseException as e:
            return False, str(e)


# ffmpeg -ss 00:00:30 -i $(youtube-dl -f 22 --get-url "$youtube_url") -vframes 1 -q:v 2 out.jpg