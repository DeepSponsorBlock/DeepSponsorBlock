import click
import pathlib
import concurrent.futures

from tqdm import tqdm

from .segments import load_segments


@click.command()
@click.option('--fps', default=1, help='Frames to extract per second.', type=float)
@click.option('--max-threads', default=20, help='Number of threads to run on.', type=int)
@click.option('--start-index', default=0, help='Index of the video to start with.', type=int)
@click.option('--limit-count', default=None, help='Maximum number of videos to download.', type=int)
@click.option('--log-ffmpeg', default=False, help='Log the output of ffmpeg.', is_flag=True)
@click.argument('input_csv', type=click.Path(exists=True, readable=True))
@click.argument('output_dir', type=click.Path(dir_okay=True, file_okay=False, writable=True))
def fetch(fps, max_threads, start_index, limit_count, log_ffmpeg, input_csv, output_dir):
    output_path = pathlib.Path(output_dir)
    # Helper function to run on each thread.
    def download(video):
        return video.download(output_path, fps, log_ffmpeg)

    # Load the segments.
    videos_to_download = load_segments(input_csv)[start_index:]

    # Limit the number of videos if needed.
    if limit_count is not None:
        videos_to_download = videos_to_download[:limit_count]

    # Now concurrently fetch them.
    if max_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(download, video) : video for video in videos_to_download}

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

            print("Completed: %d out of %d videos had errors." % (len(unsuccessful), len(futures)))
            print("Videos with errors: " + ", ".join(unsuccessful))
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

        print("Completed: %d out of %d videos had errors." % (len(unsuccessful), len(videos_to_download)))
        print("Videos with errors: " + ", ".join(unsuccessful))

if __name__ == "__main__":
    fetch(prog_name="python -m dsbfetch")