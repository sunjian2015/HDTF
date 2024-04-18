"""
This file downloads almost all the videos from the HDTF dataset. Some videos are discarded for the following reasons:
- they do not contain cropping information because they are somewhat noisy (hand moving, background changing, etc.)
- they are not available on youtube anymore (at all or in the specified format)

The discarded videos constitute a small portion of the dataset, so you can try to re-download them manually on your own.

Usage:
```
$ python download.py --output_dir /tmp/data/hdtf --num_workers 8
```

You need tqdm and youtube-dl libraries to be installed for this script to work.
"""


import os
import argparse
from typing import List, Dict
from multiprocessing import Pool
import subprocess
from subprocess import Popen, PIPE
from urllib import parse

from tqdm import tqdm


subsets = ["RD", "WDA", "WRA"]


def download_hdtf(source_dir: os.PathLike, output_dir: os.PathLike, num_workers: int, **process_video_kwargs):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '_videos_raw'), exist_ok=True)

    download_queue = construct_download_queue(source_dir, output_dir)
    task_kwargs = [dict(
        video_data=vd,
        output_dir=output_dir,
        **process_video_kwargs,
     ) for vd in download_queue]
    pool = Pool(processes=num_workers)
    tqdm_kwargs = dict(total=len(task_kwargs), desc=f'Downloading videos into {output_dir} (note: without sound)')

    for _ in tqdm(pool.imap_unordered(task_proxy, task_kwargs), **tqdm_kwargs):
        pass

    print('Download is finished, you can now (optionally) delete the following directories, since they are not needed anymore and occupy a lot of space:')
    print(' -', os.path.join(output_dir, '_videos_raw'))


def construct_download_queue(source_dir: os.PathLike, output_dir: os.PathLike) -> List[Dict]:
    download_queue = []

    for subset in subsets:
        video_urls = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_video_url.txt'))

        # # 2024-04 you-get 下载失败的视频
        # failed_videos = ['RD_Radio30.mp4', 'WRA_MitchDaniels1.mp4', 'WRA_ShelleyMooreCapito1.mp4', 'WRA_RandPaul2.mp4', 'WDA_MarkWarner.mp4', 
        #                 'WRA_FredUpton.mp4', 'WDA_ChrisCoons.mp4', 'WDA_RichardNeal1.mp4', 'WDA_TedDeutch.mp4', 'WDA_TerriSewell0.mp4', 
        #                 'RD_Radio44.mp4', 'WRA_CoryGardner.mp4', 'WRA_MarthaRoby.mp4', 'WRA_JohnKasich2.mp4', 'WDA_SheldonWhitehouse0.mp4', 
        #                 'WDA_DonnaShalala0.mp4', 'RD_Radio32.mp4', 'WDA_NancyPelosi0.mp4', 'WDA_EmanuelCleaver.mp4', 'RD_Radio47.mp4', 
        #                 'RD_Radio22.mp4', 'WRA_RandPaul1.mp4', 'WRA_GeoffDavis.mp4', 'WDA_PattyMurray1.mp4', 'RD_Radio59.mp4', 
        #                 'WRA_MarshaBlackburn1.mp4', 'WRA_CathyMcMorrisRodgers2.mp4']
        # failed_videos_name = [n.split('.')[0].split('_')[1] for n in failed_videos]
        # new_video_urls = {k:v for k, v in video_urls.items() if k in failed_videos_name}
        # video_urls = new_video_urls


        crops = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_crop_wh.txt'))
        intervals = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_annotion_time.txt'))
        resolutions = read_file_as_space_separated_data(os.path.join(source_dir, f'{subset}_resolution.txt'))

        for video_name, (video_url,) in video_urls.items():
            if not f'{video_name}.mp4' in intervals:
                print(f'Entire {subset}/{video_name} does not contain any clip intervals, hence is broken. Discarding it.')
                continue

            if not f'{video_name}.mp4' in resolutions or len(resolutions[f'{video_name}.mp4']) > 1:
                print(f'Entire {subset}/{video_name} does not contain the resolution (or it is in a bad format), hence is broken. Discarding it.')
                continue

            all_clips_intervals = [x.split('-') for x in intervals[f'{video_name}.mp4']]
            clips_crops = []
            clips_intervals = []

            for clip_idx, clip_interval in enumerate(all_clips_intervals):
                clip_name = f'{video_name}_{clip_idx}.mp4'
                if not clip_name in crops:
                    print(f'Clip {subset}/{clip_name} is not present in crops, hence is broken. Discarding it.')
                    continue
                clips_crops.append(crops[clip_name])
                clips_intervals.append(clip_interval)

            clips_crops = [list(map(int, cs)) for cs in clips_crops]

            if len(clips_crops) == 0:
                print(f'Entire {subset}/{video_name} does not contain any crops, hence is broken. Discarding it.')
                continue

            assert len(clips_intervals) == len(clips_crops)
            assert set([len(vi) for vi in clips_intervals]) == {2}, f"Broken time interval, {clips_intervals}"
            assert set([len(vc) for vc in clips_crops]) == {4}, f"Broken crops, {clips_crops}"
            assert all([vc[1] == vc[3] for vc in clips_crops]), f'Some crops are not square, {clips_crops}'

            download_queue.append({
                'name': f'{subset}_{video_name}',
                'id': parse.parse_qs(parse.urlparse(video_url).query)['v'][0],
                'intervals': clips_intervals,
                'crops': clips_crops,
                'output_dir': output_dir,
                'resolution': resolutions[f'{video_name}.mp4'][0]
            })

    return download_queue


def task_proxy(kwargs):
    return download_and_process_video(**kwargs)


def download_and_process_video(video_data: Dict, output_dir: str):
    """
    Downloads the video and cuts/crops it into several ones according to the provided time intervals
    """
    raw_download_path = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}")
    raw_download_log_file = os.path.join(output_dir, '_videos_raw', f"{video_data['name']}_download_log.txt")
    download_result = download_video(video_data['id'], raw_download_path, resolution=video_data['resolution'], log_file=raw_download_log_file)

    if not download_result:
        print('Failed to download', video_data)
        print(f'See {raw_download_log_file} for details')
        return

    for suffix in ['mp4', 'mkv', 'webm', 'wmv', 'mpg', 'mpeg', 'mov', 'avi', 'flv']:
        tmp = raw_download_path + '.' + suffix
        if os.path.isfile(tmp):
            raw_download_path = tmp
            break

    raw_download_path = os.path.abspath(raw_download_path) # 转成绝对路径
    # 如果不是mp4格式则转为mp4格式
    if not raw_download_path.endswith('.mp4'):
        print('Convert to mp4 format...')
        tmp = raw_download_path.split('.')[0] + '.mp4'
        command = f'ffmpeg -i {raw_download_path} -q 0 -vcodec libx264 -acodec aac -pixel_format yuv420p -v quiet {tmp}'
        subprocess.call(command, shell=True)
        os.remove(raw_download_path)
        raw_download_path = tmp

    # We do not know beforehand, what will be the resolution of the downloaded video
    # Youtube-dl selects a (presumably) highest one
    video_resolution = get_video_resolution(raw_download_path)
    # if not video_resolution != video_data['resolution']:
    #     print(f"Downloaded resolution is not correct for {video_data['name']}: {video_resolution} vs {video_data['name']}. Discarding this video.")
    #     return
    if video_resolution != int(video_data['resolution']):
        print('resolution is not correct, resize...')
        tmp = raw_download_path.replace('.mp4', '_tmp.mp4')
        command = f"ffmpeg -i {raw_download_path} -vf scale=-1:{video_data['resolution']} -v quiet {tmp}"
        subprocess.call(command, shell=True)
        os.remove(raw_download_path)
        raw_download_path = tmp.replace('_tmp.mp4', '.mp4')
        os.rename(tmp, raw_download_path)

    for clip_idx in range(len(video_data['intervals'])):
        start, end = video_data['intervals'][clip_idx]
        clip_name = f'{video_data["name"]}_{clip_idx:03d}'
        clip_path = os.path.join(output_dir, clip_name + '.mp4')
        crop_success = cut_and_crop_video(raw_download_path, clip_path, start, end, video_data['crops'][clip_idx])

        if not crop_success:
            print(f'Failed to cut-and-crop clip #{clip_idx}', video_data)
            continue


def read_file_as_space_separated_data(filepath: os.PathLike) -> Dict:
    """
    Reads a file as a space-separated dataframe, where the first column is the index
    """
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
        lines = [[v.strip() for v in l.strip().split(' ')] for l in lines]
        data = {l[0]: l[1:] for l in lines}

    return data


def download_video(video_id, download_path, resolution: int=None, video_format="mp4", log_file=None):
    """
    Download video from YouTube.
    :param video_id:        YouTube ID of the video.
    :param download_path:   Where to save the video.
    :param video_format:    Format to download.
    :param log_file:        Path to a log file for youtube-dl.
    :return:                Tuple: path to the downloaded video and a bool indicating success.

    Copy-pasted from https://github.com/ytdl-org/youtube-dl
    """
    # if os.path.isfile(download_path): return True # File already exists

    if log_file is None:
        stderr = subprocess.DEVNULL
    else:
        stderr = open(log_file, "a")

    # video_selection = f"bestvideo[ext={video_format}]"
    # 修复下载视频无声音 https://blog.csdn.net/jiaoyangwm/article/details/133015443
    # https://blog.csdn.net/jiaoyangwm/article/details/133015443#:~:text=2%E3%80%81-,%E4%B8%8B%E8%BD%BD,-HDTF%20%E6%95%B0%E6%8D%AE
    # video_selection = f"best[ext={video_format}]"

    # 方法一：下载失败的可以用这个方式解决
    # video_selection = f"bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"
    # video_selection = video_selection if resolution is None else f"{video_selection}[height={resolution}]"
    # command = [
    #     "youtube-dl",
    #     "https://youtube.com/watch?v={}".format(video_id), "--quiet", "-f",
    #     video_selection,
    #     "--output", download_path,
    #     "--no-continue"
    # ]

    # # 方法二：可以不用选择分辨率但下载的视频和文件中的分辨率可能不一致
    # command = [
    #     "youtube-dl",
    #     "https://youtube.com/watch?v={}".format(video_id), "--quiet",
    #     "--output", download_path,
    #     "--no-continue"
    # ]

    # 改为用 you-get 下载 - 作者推荐的方法
    # v_url = "https://youtube.com/watch?v={}".format(video_id)
    # command = f"you-get {v_url} --no-caption -O {download_path}"
    command = [
        "you-get",
        "https://youtube.com/watch?v={}".format(video_id),
        "-O", download_path,
        "--no-caption"
    ]

    return_code = subprocess.call(command, stderr=stderr)
    success = return_code == 0

    if log_file is not None:
        stderr.close()

    return success # and os.path.isfile(download_path)


def get_video_resolution(video_path: os.PathLike) -> int:
    command = ' '.join([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "csv=p=0",
        video_path
    ])

    process = Popen(command, stdout=PIPE, shell=True)
    (output, err) = process.communicate()
    return_code = process.wait()
    success = return_code == 0

    if not success:
        print('Command failed:', command)
        return -1

    return int(output)


def cut_and_crop_video(raw_video_path, output_path, start, end, crop: List[int]):
    # if os.path.isfile(output_path): return True # File already exists

    x, out_w, y, out_h = crop

    command = ' '.join([
        "ffmpeg", "-i", raw_video_path,
        "-strict", "-2", # Some legacy arguments
        "-loglevel", "quiet", # Verbosity arguments
        "-qscale", "0", # Preserve the quality
        "-y", # Overwrite if the file exists
        "-ss", str(start), "-to", str(end), # Cut arguments
        "-filter:v", f'"crop={out_w}:{out_h}:{x}:{y}"', # Crop arguments
        output_path
    ])

    return_code = subprocess.call(command, shell=True)
    success = return_code == 0

    if not success:
        print('Command failed:', command)

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HDTF dataset")
    parser.add_argument('-s', '--source_dir', type=str, default='HDTF_dataset', help='Path to the directory with the dataset')
    parser.add_argument('-o', '--output_dir', type=str, help='Where to save the videos?')
    parser.add_argument('-w', '--num_workers', type=int, default=8, help='Number of workers for downloading')
    args = parser.parse_args()

    download_hdtf(
        args.source_dir,
        args.output_dir,
        args.num_workers,
    )
