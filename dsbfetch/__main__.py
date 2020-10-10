import click
import pathlib
import concurrent.futures

from tqdm import tqdm

from .segments import load_segments


@click.command()
@click.option('--fps', default=1, help='Frames to extract per second.', type=float)
@click.option('--max-threads', default=20, help='Number of threads to run on.', type=int)
@click.option('--limit-count', default=None, help='Maximum number of videos to download.', type=int)
@click.argument('input_csv', type=click.Path(exists=True, readable=True))
@click.argument('output_dir', type=click.Path(dir_okay=True, file_okay=False, writable=True))
def fetch(fps, max_threads, limit_count, input_csv, output_dir):
    output_path = pathlib.Path(output_dir)
    # Helper function to run on each thread.
    def download(video):
        return video.download(output_path, fps)

    # Load the segments.
    v = load_segments(input_csv)

    # Limit the number of videos if needed.
    videos_to_download = list(v.values())
    if limit_count is not None:
        videos_to_download = videos_to_download[:limit_count]

    # Now concurrently fetch them.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(download, video) for video in videos_to_download]

        # Show a tqdm progress bar.
        kwargs = {
            'total': len(futures),
            'unit': 'video',
            'leave': True
        }
        for f in tqdm(concurrent.futures.as_completed(futures), **kwargs):
            pass

if __name__ == "__main__":
    fetch(prog_name="python -m dsbfetch")