# DeepSponsorBlock
Deep learning-based solution for identifying sponsored content segments on YouTube videos.

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