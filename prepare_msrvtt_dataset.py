from pathlib import Path
import pathlib
from tqdm import tqdm
from typing import List, Tuple
import json
from torchvision.datasets.utils import _get_google_drive_file_id, extract_archive
from torchvision.datasets.utils import download_file_from_google_drive
from torchvision.io.video import read_video
import av


import pytube
from pytube import YouTube


from urllib.request import HTTPError
from urllib.error import URLError
from socket import gaierror
from http.client import IncompleteRead

'''
except URLError:
    print("URLError Error: " + video_id)
    stat = "URLError Error"
    return stat, False
except gaierror:
    print("gaierror Error: " + video_id)
    stat = "gaierror Error"
    return stat, False
'''
from video_utils import clip_video

from text_processing import get_captions_length
from torchtext.data.utils import get_tokenizer

import csv

# Default number for seed is 0.
# random.seed(0)

# VAL_IDS = 6513  # index number where validation data starts.


def load_json_list(json_path: pathlib.Path, video_folder: pathlib.Path, split="train") -> Tuple[List[str], List[str], List[str]]:
    # Initialize video_paths list.
    video_paths = []
    # Initialize ids list.
    video_ids = []
    # Initialize train captions list.
    captions = []
    # Load json file in train_data
    data = json.loads(json_path.read_bytes())

    for annotation in data['info']:
        if annotation['split'] == split:
            video_name = annotation['video_id']
            video_path = video_folder / f"{video_name}.mp4"
            video_paths.append(video_path)
            video_ids.append(annotation['id'])

    # Go through train data.
    print(f'Loading {str(json_path)} data...')
    for annotation in tqdm(data['sentences']):
        caption = 'boc ' + annotation['caption'] + ' eoc'
        captions.append(caption)

    print('Data is loaded.')
    
    return video_paths, captions, video_ids

class MSRVTTDataset():
    '''
        Utilities for video captioning dataset of MSRVTT

        Initialize:
        # dt = MSRVTTDataset()

        Download MSRVTT Dataset:
        # dt.download_dataset()

        Load captions, paths, times and ids:
        # train_data, test_data = dt.load_data()

        # paths, captions, ids, start_times, end_times = zip(*train_data)
        In test data captions are not important therefore only one corresponding caption for a video added.
        # paths, captions, ids, start_times, end_times = zip(*test_data)

    '''
    def __init__(self, root_folder:pathlib.Path = None) -> None:

        # name of the dataset.
        self.name = "MSRVTT"

        # url links for the dataset train videos in zip format.
        self.train_path = ["https://drive.google.com/file/d/1XyZwkCGV2zF90jjfmkqVlWvWUVWqCr66", "train_val_videos.zip"]

        # url links for the dataset val videos in zip format.
        self.test_path = ["https://drive.google.com/file/d/17Q4Cq-QwO9ygjbVV9OBeJqGks2DwtltH", "test_videos.zip"]

        # url links for the dataset annotations in zip format.
        self.train_annotations_path = ["https://drive.google.com/file/d/1mglvNKhJ-igKiQFFk8RJfw9A8Xp1vLM6", "train_val_annotation.zip"]

        # url links for the dataset annotations in zip format.
        self.test_annotations_path = ["https://drive.google.com/file/d/16iaSq_qi3ve3coqZHokcWJCvIEv6AcH3", "test_annotation.zip"]

        # Project root Path
        self.root_folder = Path('/media/envisage/backup1/Murat/Datasets')

        # Drive Path
        self.dataset_folder = self.root_folder / self.name

        # train videos Path
        self.train_folder = self.dataset_folder / "TrainValVideo"

        # val videos Path
        self.test_folder = self.dataset_folder / "TestVideo"

        # train frames Path
        self.train_frame_folder = self.dataset_folder / 'TrainValVideoFrames'

        # train frames Path
        self.test_frame_folder = self.dataset_folder / 'TestVideoFrames'

        # initial train captions path
        self.default_train_annotations = self.dataset_folder / "train_val_videodatainfo.json"

        # initial test captions path
        self.default_test_annotations = self.dataset_folder / "test_videodatainfo.json"

        # updated train captions path
        self.train_annotations = self.dataset_folder / "train_annotations.json"

        # updated train captions path
        self.val_annotations = self.dataset_folder / "val_annotations.json"

        # updated train captions path
        self.test_annotations = self.dataset_folder / "test_annotations.json"

        # train visual features Path
        self.train_visual_features_folder = self.dataset_folder / "features_visual_train"

        # train audial features Path
        self.train_audial_features_folder = self.dataset_folder / "features_audial_train"

        # test features Path
        self.test_visual_features_folder = self.dataset_folder / "features_visual_test"

        # test features Path
        self.test_audial_features_folder = self.dataset_folder / "features_audial_test"

        # info folder Path
        self.info_folder = self.dataset_folder / "info"

        self.models_folder = self.dataset_folder / "models"

        self.tensorboard_folder = self.dataset_folder / "tensorboard_files"

        # Min and Max caption lengths which cover most of the captions. 80%
        self.min_caption_length = 4
        self.max_caption_length = 12

        # Max frames sizes for videos factored.
        self.max_frames = {
            "1": {
                "audio": 258,
                "visual": 901
            },
            "0.9": {
                "audio": 232,
                "visual": 811
            },
            "0.8": {
                "audio": 207,
                "visual": 721
            },
            "0.7": {
                "audio": 181,
                "visual": 631
            },
            "0.6": {
                "audio": 155,
                "visual": 541
            },
            "0.5": {
                "audio": 129,
                "visual": 450
            },
            "0.4": {
                "audio": 103,
                "visual": 360
            }
        }

    def download_annotations(self) -> None:
        '''
            Download the dataset's train videos, test videos and their annotations into the MSRVTT folder.
        '''

        # dataset paths in a list for downloading from google drive.
        # download_list = [self.train_path, self.test_path, self.train_annotations_path, self.test_annotations_path]
        download_list = [self.train_annotations_path, self.test_annotations_path]
        save_list = [self.default_train_annotations, self.default_test_annotations]

        for download_path, save_path in zip(download_list, save_list):
            # check if the annotation file is already downloaded. if true do not download it again and continue.
            if save_path.exists():
                print(f'{save_path.name} already exists.')
                continue
            # check if the dataset folder exists if not create it.
            if not self.dataset_folder.exists():
                Path(self.dataset_folder).mkdir(parents=True, exist_ok=True)
            
            # download_and_extract_archive(url=path, download_root=str(self.drive_data_zip_paths), extract_root=str(self.dataset_folder))
            file_id = _get_google_drive_file_id(download_path[0])
            file = self.dataset_folder / download_path[1]
            download_file_from_google_drive(file_id=file_id, root=str(self.dataset_folder), filename=download_path[1])
            extract_archive(str(file), str(self.dataset_folder))
            file.unlink()

    def download_videos(self) -> None:
        '''
            Download videos from youtube into the MSRVTT folder.
        '''

        download_list = [self.default_train_annotations, self.default_test_annotations]
        download_folder_list = [self.train_folder, self.test_folder]

        for path, folder in zip(download_list, download_folder_list):
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

            status = []
            
            with open(path, 'r') as json_path:
                data = json.load(json_path)
                for annotation in data['videos']:
                    video_id = annotation['video_id']
                    video_name = video_id + '.mp4'
                    start_time = annotation['start time']
                    end_time = annotation['end time']
                    url = annotation['url']
                    stat, is_available = self.download_video(video_id=video_id, start_time=start_time, end_time=end_time, url=url, download_folder=folder)


                    status.append({'video_id': video_id, 'start_time': start_time, 'end_time': end_time, 'status': stat, 'is_available': is_available, 'url': url})
                
            # write status to json file.
            with open(str(folder.parent / f'{folder.name}_status.json'), 'w') as f:
                json.dump(status, f)

    # download youtube videos start time to end time from id.
    def download_video(self, video_id, start_time, end_time, url, download_folder) -> None:

        '''
            Download youtube videos start time to end time from id.
        '''

        yt = YouTube(url)
        try:
            download_video_path = download_folder / video_id
            clipped_video_path = download_video_path.with_suffix('.mp4')
            if not clipped_video_path.exists():
                yt = yt.streams.filter(file_extension="mp4", resolution="360p").first().download(output_path=str(download_folder), filename=video_id)
                clip_video(video_path=download_video_path, start_time=start_time, end_time=end_time, output_path=str(clipped_video_path))
                download_video_path.unlink()
            print("Downloaded: " + video_id)
            stat = "Available"
            return stat, True
        except pytube.exceptions.VideoUnavailable:
            print("Video Unavailable: " + video_id)
            stat = "Unavailable"
            return stat, False
        except KeyError:
            print("Key Error: " + video_id)
            stat = "Key Error"
            return stat, False
        except HTTPError:
            print("HTTP Error: " + video_id)
            stat = "HTTP Error"
            return stat, False
        except IncompleteRead:
            print("Incomplete Read: " + video_id)
            stat = "Incomplete Read Error"
            download_video_path.unlink()
            return stat, False
        except av.error.ValueError:
            print("Value Error: " + video_id)
            stat = "Value Error"
            return stat, False
      
        
    
    def download_dataset(self) -> None:
        self.download_annotations()
        self.download_videos()
        self.update_annotations()

    def update_annotations(self) -> None:

        # Train Val annotations
        # get video ids from trainval annotations
        print("Updating Train Val annotations...")
        train_video_ids = []
        train_ids = []
        val_video_ids = []
        val_ids = []
        train_annotations = []
        val_annotations = []
        with open(self.default_train_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)

            for video in data['videos']:
                if video['split'] == 'train':
                    train_video_ids.append(video['video_id'])
                    train_ids.append(video['id'])
                elif video['split'] == 'validate':
                    val_video_ids.append(video['video_id'])
                    val_ids.append(video['id'])

            # update annotations if the corresponding video exists.
            train_sen_ids = []
            val_sen_ids = []
            for sentence_data in data['sentences']:
                if sentence_data['video_id'] in train_video_ids:
                    sentence_id = sentence_data['sen_id']
                    video_name = sentence_data['video_id']
                    video_path = self.train_folder / f"{video_name}.mp4"
                    if not video_path.exists():
                        continue
                    caption = sentence_data['caption']
                    video_id = int(video_name[5:])
                    train_data = {'video_id': video_id, 'video_name': video_name, 'caption': caption, 'sentence_id': sentence_id}
                    train_annotations.append(train_data)
                elif sentence_data['video_id'] in val_video_ids:
                    sentence_id = sentence_data['sen_id']
                    video_name = sentence_data['video_id']
                    video_path = self.train_folder / f"{video_name}.mp4"
                    if not video_path.exists():
                        continue
                    caption = sentence_data['caption']
                    video_id = int(video_name[5:])
                    val_data = {'video_id': video_id, 'video_name': video_name, 'caption': caption, 'sentence_id': sentence_id}
                    val_annotations.append(val_data)
        
        # write train annotations to json file.
        with open(self.train_annotations, 'w') as f:
            json.dump(train_annotations, f)

        # write val annotations to json file.
        with open(self.val_annotations, 'w') as f:
            json.dump(val_annotations, f)

        # Test annotations
        print("Updating Test annotations...")

        with open(self.default_test_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)
            test_annotations = []
            # Check if the sentence id is repeated. Because there are some duplicate sentences in the test set. I don't know why.
            sen_ids = []
            # update annotations if the corresponding video exists.

            for sentence_data in data['sentences']:
                sentence_id = sentence_data['sen_id']
                video_name = sentence_data['video_id']
                video_path = self.test_folder / f"{video_name}.mp4"
                if not video_path.exists():
                    continue
                caption = sentence_data['caption']
                video_id = int(video_name[5:])
                test_data = {'video_id': video_id, 'video_name': video_name, 'caption': caption, 'sentence_id': sentence_id}
                test_annotations.append(test_data)

        # write test annotations to json file.
        with open(self.test_annotations, 'w') as f:
            json.dump(test_annotations, f)

    def load_annotations(self, annotations) -> Tuple[List[str], List[str], List[str]]:

        data = json.loads(annotations.read_bytes())
        names = []
        captions = []
        ids = []
        for sample in data:
            names.append(sample["video_name"])
            captions.append(sample["caption"])
            ids.append(sample["video_id"])

        return names, captions, ids

    def load_data(self) -> Tuple[List[str], List[str], List[str]]:
        '''
            Load the MSRVTT captions and their corresponding video ids.
            paths, captions, ids, start_times, end_times
        '''
        train_names, train_captions, train_ids = self.load_annotations(self.train_annotations)
        train_data = zip(train_names, train_captions, train_ids)
        val_names, val_captions, val_ids = self.load_annotations(self.val_annotations)
        val_data = zip(val_names, val_captions, val_ids)

        train_val_names = train_names + val_names
        train_val_captions = train_captions + val_captions
        train_val_ids = train_ids + val_ids

        
        train_val_data = zip(train_val_names, train_val_captions, train_val_ids)
        test_names, test_captions, test_ids = self.load_annotations(self.test_annotations)
        test_data = zip(test_names, test_captions, test_ids)

        return train_val_data, None, test_data
        #return train_data, val_data, test_data

    def get_info_of_dataset(self) -> None:
        
        if not self.info_folder.exists():
            self.info_folder.mkdir()

        caption_lengths = self.caption_lengths()
        train_video_lengths, val_video_lengths, test_video_lengths = self.video_lengths()

        # write caption_lengths to a csv file.
        with open((self.info_folder / "caption_lengths.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Length', 'Caption Lengths'])
            for length, caption_length in enumerate(caption_lengths):
                writer.writerow([length, caption_length])

        # write train video lengths to a csv file.
        with open((self.info_folder / "train_video_lengths.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Train Video Lengths'])
            for train_video_length in train_video_lengths:
                writer.writerow([train_video_length['video_id'], train_video_length['video_length']])

        # write val video lengths to a csv file.
        with open((self.info_folder / "val_video_lengths.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Val Video Lengths'])
            for val_video_length in val_video_lengths:
                writer.writerow([val_video_length['video_id'], val_video_length['video_length']])

        # write test video lengths to a csv file.
        with open((self.info_folder / "test_video_lengths.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Test Video Lengths'])
            for test_video_length in test_video_lengths:
                writer.writerow([test_video_length['video_id'], test_video_length['video_length']])

    def caption_lengths(self) -> None:
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        print('Analyzing caption lengths of the dataset if there is a corresponding video...')
        captions = []
        with open(self.train_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)

            for element in data:
                caption = element['caption']
                video_name = element['video_name']
                # video_path = self.train_folder / f"{video_name}.mp4"
                # if not video_path.exists():
                #    continue
                captions.append(caption)
        
        caption_lengths = get_captions_length(captions, tokenizer)
        return caption_lengths

    def video_lengths(self) -> None:

        print('Analyzing video lengths of the dataset if there is a corresponding video...')
        video_lengths = []
        with open(self.default_train_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)

            train_video_lengths = []
            val_video_lengths = []

            for video in data['videos']:
                video_id = video['id']
                start_time = video['start time']
                end_time = video['end time']
                video_length = round(end_time - start_time, 2)

                stat = {'video_id': video_id, 'video_length': video_length}
                if video['split'] == 'train':
                    train_video_lengths.append(stat)
                elif video['split'] == 'validate':
                    val_video_lengths.append(stat)

        
        with open(self.default_test_annotations, 'r') as json_path:
            # load json file
            data = json.load(json_path)

            test_video_lengths = []

            for video in data['videos']:
                video_id = video['id']
                start_time = video['start time']
                end_time = video['end time']
                video_length = end_time - start_time

                stat = {'video_id': video_id, 'video_length': video_length}
                test_video_lengths.append(stat)

        return train_video_lengths, val_video_lengths, test_video_lengths

    def check_video(self) -> None:

        train_videos = self.train_folder.glob('*.mp4')
        test_videos = self.test_folder.glob('*.mp4')
        for video in tqdm(train_videos):
            v, a, t = read_video(str(video))
            try:
                fps = t['audio_fps']
            except KeyError:
                print(f'{video} has no audio data in train folder.')

        # check test videos
        for video in tqdm(test_videos):
            v, a, t = read_video(str(video))
            try:
                fps = t['audio_fps']
            except KeyError:
                print(f'{video} has no audio data in test folder.')
