from pathlib import Path
import pathlib
from typing import List, Tuple
from torchvision.datasets.utils import _get_google_drive_file_id
from torchvision.datasets.utils import download_file_from_google_drive
import pytube
from pytube import YouTube
from urllib.request import HTTPError
from video_utils import clip_video
import json
import av
from urllib.error import URLError
from socket import gaierror
from http.client import IncompleteRead
from tqdm import tqdm

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
def load_list(path: pathlib.Path, video_folder: pathlib.Path, update_annotations, save_path) -> Tuple[Tuple[List[str], List[str], List[str]], Tuple[List[str], List[str], List[str]]]:
    # Initialize video_paths list.
    train_video_paths = []
    test_video_paths = []
    # Initialize ids list.
    train_video_ids = []
    test_video_ids = []
    # Initialize train captions list.
    train_captions = []
    test_captions = []

    with open(str(path)) as f:
        annotations = f.readlines()
        annotations = annotations[7:]
 
    ids = []
    downloaded_video_paths = video_folder.glob("*.*")
    for i, path in enumerate(downloaded_video_paths):
        ids.append(path.stem)
    
    
    split_index = int(len(ids) * 0.7)
    train_set_ids = ids[:split_index]
    test_set_ids = ids[split_index:]
    

    for id in tqdm(train_set_ids):
        video_name = id + ".mp4"
        video_path = video_folder / video_name
        for ann in annotations:
            if id == ann.split()[0]:
                train_video_paths.append(video_path)
                train_video_ids.append(id)
                caption = " ".join(ann.split()[1:])
                train_captions.append(caption)

    for id in tqdm(test_set_ids):
        video_name = id + ".mp4"
        video_path = video_folder / video_name
        test_annotations = []
        for ann in annotations:
            if id == ann.split()[0]:
                test_video_paths.append(video_path)
                test_video_ids.append(id)
                caption = " ".join(ann.split()[1:])
                test_captions.append(caption)

                test_data = {'video_id': id, 'video_name': video_path.name, 'caption': caption}
                test_annotations.append(test_data)

    if update_annotations:
        with open(save_path, 'w') as f:
            json.dump(test_annotations, f)




    
    return train_video_paths, train_captions, train_video_ids, test_video_paths, test_captions, test_video_ids 

class MSVDDataset():
    '''
        Utilities for video captioning dataset of MSVD

        Initialize:
        # dt = MSVDDataset()

        Download MSVD Dataset:
        # dt.download_dataset()

        Load captions, paths, times and ids:
        # train_data, test_data = dt.load_data()

        # paths, captions, ids, start_times, end_times = zip(*train_data)
        In test data captions are not important therefore only one corresponding caption for a video added.
        # paths, captions, ids, start_times, end_times = zip(*test_data)

    '''
    def __init__(self, root_folder:pathlib.Path = None) -> None:

        # name of the dataset.
        self.name = "MSVD"

        # url links for the dataset annotations in zip format.
        # self.videos = ["https://drive.google.com/file/d/11IBEBbAOHFq0rIiWBlaFvqMeqErnGzNG", "YouTubeClips.tar"]

        # url links for the dataset train videos in zip format.
        self.annotations_path = ["https://drive.google.com/file/d/1T7MOk1YPk8v1LUz3FjzwuzGX40lLHPuB", "AllVideoDescriptions.txt"]

        # Project root Path
        self.root_folder = Path('/media/envisage/backup1/Emir/Datasets')

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

        # train visual features Path
        self.train_visual_features_folder = self.dataset_folder / "features_visual_train"

        # train audial features Path
        self.train_audial_features_folder = self.dataset_folder / "features_audial_train"

        # test features Path
        self.test_visual_features_folder = self.dataset_folder / "features_visual_test"

        # test features Path
        self.test_audial_features_folder = self.dataset_folder / "features_audial_test"

        # description path
        self.annotations = self.dataset_folder / "AllVideoDescriptions.txt"
        # train captions path
        self.train_annotations = self.dataset_folder / "train_annotations.json"
        self.val_annotations = None

        # test captions path
        self.test_annotations = self.dataset_folder / "test_annotations.json"

        self.max_visual_frame_length = 1801
        self.max_visual_seq_length = 240
        self.max_audial_frame_length = 2648064
        self.max_audial_seq_length = 517
        self.min_frames_audio = 83968

        self.min_caption_length = 3
        self.max_caption_length = 12

    def download_annotations(self) -> None:
        '''
            Download the dataset's train videos, test videos and their annotations into the MSRVTT folder.
        '''

        # dataset paths in a list.
        # download_list = [self.videos, self.annotations_path]
        download_list = [self.annotations_path]

        for path in download_list:
            file = self.dataset_folder / path[1]
            if file.exists():
                print(f"{file.name} already exists.")
                continue
            # check if the dataset folder exists if not create it.
            if not self.dataset_folder.exists():
                Path(self.dataset_folder).mkdir(parents=True, exist_ok=True)
            
            # download_and_extract_archive(url=path, download_root=str(self.drive_data_zip_paths), extract_root=str(self.dataset_folder))
            file_id = _get_google_drive_file_id(path[0])
            #file = self.dataset_folder / path[1]
            download_file_from_google_drive(file_id=file_id, root=str(self.dataset_folder), filename=path[1])
            #extract_archive(str(file), str(self.dataset_folder))
            #file.unlink()

    def download_videos(self) -> None:
        '''
            Read txt file and download videos from youtube
        '''
        annotation_path = self.dataset_folder / self.annotations_path[1]
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        annotations = lines[7:]
        video_ids = []
        for annotation in annotations:
            video_id = annotation.split(" ")[0]
            video_ids.append(video_id)
        video_ids = list(set(video_ids))
        
        status = []

        for video_id in video_ids:
            start_time = float(video_id[12:].split("_")[0])
            end_time = float(video_id[12:].split("_")[1])
            youtube_id = video_id[:11]

            stat, is_available = self.download_video(video_id, start_time, end_time, self.train_folder)
            status.append({'video_id': youtube_id, 'start_time': start_time, 'end_time': end_time, 'status': stat, 'is_available': is_available})
            
        # write status to json file.
        with open(str(self.dataset_folder / 'download_status.json'), 'w') as f:
            json.dump(status, f)

    # download youtube videos start time to end time from id.
    def download_video(self, video_id, start_time, end_time, download_folder) -> None:

        '''
            Download youtube videos start time to end time from id.
        '''

        # youtube video url.
        url = "https://www.youtube.com/watch?v=" + video_id

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
            Load the MSVD captions and their corresponding video ids.
            paths, captions, ids, start_times, end_times
        '''

        train_names, train_captions, train_ids = self.load_annotations(self.train_annotations)
        train_data = zip(train_names, train_captions, train_ids)
        val_data = None
        test_names, test_captions, test_ids = self.load_annotations(self.test_annotations)
        test_data = zip(test_names, test_captions, test_ids)

        return train_data, val_data, test_data

    def update_annotations(self):
        # Initialize video_paths list.
        train_video_paths = []
        test_video_paths = []
        # Initialize ids list.
        train_video_ids = []
        test_video_ids = []
        # Initialize train captions list.
        train_captions = []
        test_captions = []

        with open(self.annotations) as f:
            annotations = f.readlines()
            annotations = annotations[7:]
    
        ids = []
        downloaded_video_paths = self.train_folder.glob("*.*")
        for i, path in enumerate(downloaded_video_paths):
            ids.append(path.stem)
        
        
        split_index = int(len(ids) * 0.7)
        train_set_ids = ids[:split_index]
        test_set_ids = ids[split_index:]
        
        train_annotations = []
        test_annotations = []

        for id in tqdm(train_set_ids):
            video_name = id + ".mp4"
            video_path = self.train_folder / video_name
            for ann in annotations:
                if id == ann.split()[0]:
                    train_video_paths.append(video_path)
                    train_video_ids.append(id)
                    caption = " ".join(ann.split()[1:])
                    train_captions.append(caption)

                    train_data = {'video_id': id, 'video_name': video_path.name, 'caption': caption}
                    train_annotations.append(train_data)

        for id in tqdm(test_set_ids):
            video_name = id + ".mp4"
            video_path = self.train_folder / video_name
            for ann in annotations:
                if id == ann.split()[0]:
                    test_video_paths.append(video_path)
                    test_video_ids.append(id)
                    caption = " ".join(ann.split()[1:])
                    test_captions.append(caption)

                    test_data = {'video_id': id, 'video_name': video_path.name, 'caption': caption}
                    test_annotations.append(test_data)

        with open(self.train_annotations, 'w') as f:
            json.dump(train_annotations, f)

        with open(self.test_annotations, 'w') as f:
            json.dump(test_annotations, f)
