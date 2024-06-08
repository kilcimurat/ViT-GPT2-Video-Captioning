# TODO tokenize captions first and then feed to the class Dataset.
# TODO document the processings and have comments in codes.
import torch

from text_processing import get_vocab, preprocess_txt

# Parameters
# Eva:
# PARAMS = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 16}

# ozkan lab computer:
PARAMS = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 6}

class TestDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, paths, ids, feature_folder):
        'Initialization'
        self.paths = paths
        self.paths = list(set(paths))
        self.ids = ids
        self.feature_folder = feature_folder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.paths[index]
        #id = self.paths[index].replace(".mp4", "")
        id = self.ids[index]
        FRAME_LEN = 16
        FEATURE_LEN = 768
        feature = torch.zeros((FRAME_LEN, FEATURE_LEN))
        for i in range(FRAME_LEN):
            if "video" in name:
                name = f"video{id}_frame_{i}.pt"
            else:
                name = f"{id}_frame_{i}.pt"
            frame_feature = torch.load(self.feature_folder / name, map_location='cpu')
            feature[i] = frame_feature

        return feature, id

class TrainDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, paths, ids, captions, tokenizer, feature_folder, max_length=30):
        'Initialization'
        
        self.paths = paths
        #self.paths = list(set(paths))
        self.ids = ids
        self.captions = captions
        self.tokenizer = tokenizer
        self.feature_folder = feature_folder
        self.max_length = max_length

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.paths[index]
        #id = self.paths[index].replace(".mp4", "")
        id = self.ids[index]

        FRAME_LEN = 16
        FEATURE_LEN = 768
        feature = torch.zeros((FRAME_LEN, FEATURE_LEN))
        for i in range(FRAME_LEN):
            if "video" in name:
                name = f"video{id}_frame_{i}.pt"
            else:
                name = f"{id}_frame_{i}.pt"
            frame_feature = torch.load(self.feature_folder / name, map_location='cpu')
            feature[i] = frame_feature
        

        caption = self.captions[index]
        start_token = self.tokenizer.bos_token
        end_token = self.tokenizer.eos_token
        encoding = self.tokenizer.encode_plus(
            f"{start_token} {caption} {end_token}",
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # The 'input_ids' are the tokenized representation of the caption
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)  # Remove batch dimension
        #attention_mask = torch.ones(feature.shape[0], dtype=torch.long)
        return feature, input_ids, attention_mask

def get_loader_and_vocab(dt, tokenizer, train_visual_features_folder, test_visual_features_folder):
    train_data, val_data, test_data = dt.load_data()
    train_paths, train_captions, train_ids = zip(*train_data)
    train_dataset = TrainDataset(train_paths, train_ids, train_captions, tokenizer, feature_folder=train_visual_features_folder)
    train_loader = torch.utils.data.DataLoader(train_dataset, **PARAMS)
   

  
    val_loader = None


    test_names, test_captions, test_ids = zip(*test_data)
    test_ids = list(set(test_ids))
    test_dataset = TestDataset(paths=test_names, ids=test_ids, feature_folder=test_visual_features_folder)
    test_loader = torch.utils.data.DataLoader(test_dataset, **PARAMS)
    return train_loader, val_loader, test_loader

