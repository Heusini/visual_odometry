import numpy as np
from pathlib import Path
from enum import Enum
import cv2
import os

class Dataset(Enum):
    KITTI = 0
    MALAGA = 1
    PARKING = 2

dataset_root = Path(__file__).parents[1].joinpath("data")
paths = {
    Dataset.KITTI: dataset_root.joinpath("kitti"),
    Dataset.MALAGA: dataset_root.joinpath("malaga-urban-dataset-extract-07"),
    Dataset.PARKING: dataset_root.joinpath("parking"),
}

class DataLoader:

    def __init__(self, dataset):
        self.dataset = dataset
        self.data_path = paths[dataset]
        self.k = None
        self.poses = None
        self.images = None

    def load_data(self):
        self._load_k()
        self._load_frames()
        self._load_poses()

    def _load_k(self):
        method = f'_load_{self.dataset.name.lower()}_k'
        getattr(self, method)()

    def _load_frames(self):
        method = f'_load_{self.dataset.name.lower()}_frames'
        getattr(self, method)(self.data_path)

    def _load_poses(self):
        method = f'_load_{self.dataset.name.lower()}_poses'
        getattr(self, method)(self.data_path)

    def _load_kitti_k(self):
        # Load KITTI specific 'k' data
        self.k = np.array([[7.188560000000e+02, 0, 6.071928000000e+02],
                           [0, 7.188560000000e+02, 1.852157000000e+02],
                           [0, 0, 1]])

    def _load_malaga_k(self):
        # Load MALAGA specific 'k' data
        self.k = np.array([[621.18428, 0, 404.0076],
                           [0, 621.18428, 309.05989],
                           [0, 0, 1]])

    def _load_parking_k(self):
        # Load PARKING specific 'k' data
        self.k = np.array([[331.37, 0, 320],
                           [0, 369.568, 240],
                           [0, 0, 1]])

    def _load_kitti_frames(self, path):
        # Load KITTI specific frames
        frames_path = path.joinpath("05/image_0")
        frames = []
        for filename in os.listdir(frames_path):
            if filename.endswith('.png'):
                img = cv2.imread(str(frames_path.joinpath(filename)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frames.append(img)
        self.images = np.array(frames)

    def _load_malaga_frames(self, path):
        # Load MALAGA specific frames
        frames_path = path.joinpath("images")
        frames = []
        for filename in os.listdir(frames_path):
            if filename.endswith('left.jpg'):
                img = cv2.imread(str(frames_path.joinpath(filename)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frames.append(img)
        self.images = np.array(frames)

    def _load_parking_frames(self, path):
        # Load PARKING specific frames
        frames_path = path.joinpath("images")
        frames = []
        for filename in os.listdir(frames_path):
            if filename.endswith('.png'):
                img = cv2.imread(str(frames_path.joinpath(filename)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frames.append(img)
        self.images = np.array(frames)

    def _load_kitti_poses(self, path):
        # Load KITTI specific poses
        poses_path = path.joinpath("poses/05.txt")
        poses = np.loadtxt(poses_path)
        self.poses = np.array([np.append(np.reshape(pose, (3, 4)), np.array([[0, 0, 0, 1]]), axis=0) for pose in poses], dtype=np.float32)


    def _load_malaga_poses(self, path):
        return None

    def _load_parking_poses(self, path):
        # Load PARKING specific poses
        poses_path = path.joinpath("poses.txt")
        poses = np.loadtxt(poses_path)
        self.poses = np.array([np.append(np.reshape(pose, (3, 4)), np.array([[0, 0, 0, 1]]), axis=0) for pose in poses], dtype=np.float32)


    def get_data(self):
        return self.k, self.poses, self.images

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Initialize DataLoader with a dataset
    loader = DataLoader(Dataset.MALAGA)

    # Load the data
    loader.load_data()

    # Get the data
    k, poses, images = loader.get_data()

    # Print the 'k' values
    print("K values:")
    print(k)

    # Print the poses
    print("Poses:")
    print(poses)

    # Display the first image
    plt.imshow(images[0], cmap='gray')
    plt.show()