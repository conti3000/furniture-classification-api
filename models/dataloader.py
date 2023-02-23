from torch.utils.data import Dataset
from PIL import Image
import os 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class BaseDataset(Dataset):
    def __init__(self,img_paths, labels, transforms = None):
        self.image_paths = img_paths
        self.labels = labels
        self.transforms = transforms
                
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)
        if self.transforms is not None:
          image = self.transforms(image)

        return image, label

class FurnitureDataset():
    def __init__(self, data_dir, transforms, split=0.8):
        self.data_dir = data_dir
        self.image_paths = []
        self.labels = []
        self.transforms = transforms
        for label in os.listdir(data_dir):
            if not label.startswith('.'):
              for file_name in os.listdir(os.path.join(data_dir, label)):
                  self.image_paths.append(os.path.join(data_dir, label, file_name))
                  self.labels.append(label)

        self.image_paths_train, self.image_paths_test, self.train_labels, self.test_labels = train_test_split(
            self.image_paths, self.labels, test_size=0.33)
        
        print(len(self.image_paths_test))
        print(len(self.test_labels))

    def get_datasets(self):
        train_dataset = BaseDataset(self.image_paths_train, self.train_labels, self.transforms)
        test_dataset =  BaseDataset(self.image_paths_test, self.test_labels, self.transforms)
        
        return {"train":train_dataset, "test":test_dataset}
        