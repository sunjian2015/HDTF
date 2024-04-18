import os
import json
from tqdm import tqdm

video_dir = '_videos_raw'
download_logs = [n for n in os.listdir(video_dir) if n.endswith('.txt')]
failed = []
for s in download_logs:
    mp4_name = s.split('.')[0].replace('_download_log', '') + '.mp4'
    if not os.path.exists(os.path.join(video_dir, mp4_name)):
        failed.append(mp4_name)
print(failed)