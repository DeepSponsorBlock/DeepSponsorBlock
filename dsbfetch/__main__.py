import pathlib
import concurrent.futures
import random
import tempfile

import click
from tqdm import tqdm

from .predict import predict as predict_fn
from .predict import get_youtube_link
from .segments import load_segments


@click.group()
def cli():
    pass

@cli.command()
@click.option('--fps', default=1,
              help='Frames to extract per second.', type=float)
@click.option('--max-threads', default=20,
              help='Number of threads to run on.', type=int)
@click.option('--start-index', default=0,
              help='Index of the video to start with.', type=int)
@click.option('--limit-count', default=None,
              help='Maximum number of videos to download.', type=int)
@click.option('--log-ffmpeg', default=False,
              help='Log the output of ffmpeg.', is_flag=True)
@click.option('--shuffle', default=False,
              help='Shuffle the video order instead of using the database order.', is_flag=True)
@click.argument('input_csv',
                type=click.Path(exists=True, readable=True))
@click.argument('output_dir',
                type=click.Path(dir_okay=True, file_okay=False, writable=True))
def fetch(fps, max_threads, start_index, limit_count, log_ffmpeg, shuffle,
          input_csv, output_dir):
    output_path = pathlib.Path(output_dir)
    # Helper function to run on each thread.
    def download(video):
        return video.download(output_path, fps, log_ffmpeg)

    def check(video):
        return video.is_already_downloaded(output_path)

    # Load the segments.
    videos_to_download = load_segments(input_csv)[start_index:]

    # Shuffle if needed.
    if shuffle:
        random.shuffle(videos_to_download)

    # Limit the number of videos if needed.
    if limit_count is not None:
        videos_to_download = videos_to_download[:limit_count]

    # Filter the videos by whether or not they are already downloaded.
    click.echo("Checking already downloaded videos.")
    prefilter_count = len(videos_to_download)
    filtered_videos = []

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_threads) as executor:
        futures = {executor.submit(check, video): video
                   for video in videos_to_download}

        # Show a tqdm progress bar.
        kwargs = {
            'total': len(futures),
            'unit': 'video',
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures.keys()), **kwargs):
            if not f.result():
                filtered_videos.append(futures[f])

    videos_to_download = filtered_videos
    postfilter_count = len(videos_to_download)
    click.echo("%d/%d videos already downloaded. %d remaining." %
          (prefilter_count - postfilter_count, prefilter_count, postfilter_count))

    # Now concurrently fetch them.
    click.echo("\nStarting download.")
    if max_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(download, video) : video
                       for video in videos_to_download}

            unsuccessful = []

            # Show a tqdm progress bar.
            kwargs = {
                'total': len(futures),
                'unit': 'video',
                'leave': True
            }
            for f in tqdm(concurrent.futures.as_completed(futures.keys()), **kwargs):
                status, _ = f.result()
                if not status:
                    unsuccessful.append(futures[f].video_id)

            click.echo("Completed: %d out of %d videos had errors." % (
                len(unsuccessful), len(futures)))
            click.echo("Videos with errors: " + ", ".join(unsuccessful))
    else:
        unsuccessful = []

        kwargs = {
            'total': len(videos_to_download),
            'unit': 'video',
            'leave': True
        }
        for v, (status, _) in tqdm(((v, download(v)) for v in videos_to_download), **kwargs):
            if not status:
                unsuccessful.append(v.video_id)

        click.echo("Completed: %d out of %d videos had errors." % (
            len(unsuccessful), len(videos_to_download)))
        click.echo("Videos with errors: " + ", ".join(unsuccessful))

@cli.command()
@click.option('--batch-size', default=1024,
              help='Batch size when iterating over video frames.', type=int)
@click.option('--verbose', '-v', default=False,
              help='Print process step-by-step.', is_flag=True)
@click.argument('video_id', type=str)
def predict(video_id, batch_size, verbose):
    # Create temporary directory for frames.
    with tempfile.TemporaryDirectory() as root_path:
        start, end = predict_fn(
            video_id, pathlib.Path(root_path), batch_size=batch_size,
            verbose=verbose)
        url = get_youtube_link(video_id, start, end)
        click.echo("\nPrediction successful. Segment link: %s" % url)

if __name__ == "__main__":
    cli(prog_name="python -m dsbfetch")