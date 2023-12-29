import numpy as np
from pathlib import Path
from enum import Enum
import cv2
import os
from utils.path_loader import PathLoader
from state import FrameState

class Dataset(Enum):
    KITTI = 0
    MALAGA = 1
    PARKING = 2
    WOKO = 3

dataset_root = "../data/"

paths = {
    Dataset.KITTI: dataset_root + "kitti/",
    Dataset.MALAGA: dataset_root + "malaga-urban-dataset-extract-07/",
    Dataset.PARKING: dataset_root + "parking/",
    Dataset.WOKO: dataset_root + "woko_dataset/"
}

class DataLoader:

    best_params = {
        "KITTI": {"init_frame_indices": [0, 3]},
        "PARKING": {"init_frame_indices": [0, 3]},
        "MALAGA": {"init_frame_indices": [0, 4]},
        "WOKO": {"init_frame_indices": [0, 3]},
    }
    def __init__(self, dataset, start=0, stride=1, steps=float('inf')):
        self.dataset = dataset
        self.data_path = paths[dataset]
        self.k = None
        self.poses = None
        self.states = []
        self.start = start
        self.stride = stride
        self.steps = steps

    def __len__(self):
        if self.states is not None:
            return len(self.states)
        return 0

    def load_data(self):
        self._load_k()
        self._load_frames()
        self._load_poses()
        self._load_params()

    def _load_k(self):
        method = f'_load_{self.dataset.name.lower()}_k'
        getattr(self, method)()

    def _load_frames(self):
        method = f'_load_{self.dataset.name.lower()}_frames'
        getattr(self, method)(self.data_path)

    def _load_poses(self):
        method = f'_load_{self.dataset.name.lower()}_poses'
        getattr(self, method)(self.data_path)

    def _load_params(self):
        method = f'_load_{self.dataset.name.lower()}_poses'
        getattr(self, method)(self.data_path)

    def _load_kitti_params(self):
        # TODO: Add params specific for kitti dataset
        pass

    def _load_parking_params(self):
        # TODO: Add params specific for parking dataset
        pass

    def _load_malaga_params(self):
        # TODO: Add params specific for malaga dataset
        pass

    def _load_woko_params(self):
        # TODO: Add params specific for woko dataset
        pass

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
    
    def _load_woko_k(self):
        # Load WOKO specific 'k' data
        self.k = np.array([
            [1.80347578e+04, 0.00000000e+00, 3.64691295e+02],
            [0.00000000e+00, 4.32128296e+02, 2.06617693e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])

    def _load_kitti_frames(self, path):
        # Load KITTI specific frames
        frames_path = path + "05/image_0/"
        path_loader = PathLoader(frames_path, start=self.start, stride=self.stride)
        path_iter = iter(path_loader)

        img_path, stop = next(path_iter)
        i = 0
        while not stop:
            if i >= self.steps:
                break
            self.states += [FrameState(i, img_path)]
            img_path, stop = next(path_iter)
            i += 1
        
    def _load_malaga_frames(self, path):
        # Load MALAGA specific frames
        frames_path = path + "Images/"
        path_loader = PathLoader(frames_path, start=self.start, stride=self.stride, filter=self._malaga_file_filter)
        path_iter = iter(path_loader)

        img_path, stop = next(path_iter)
        i = 0
        while not stop:
            if i >= self.steps:
                break
            self.states += [FrameState(i, img_path)]
            img_path, stop = next(path_iter)
            i += 1

    def _load_parking_frames(self, path):
        # Load PARKING specific frames
        frames_path = path + "images/"
        path_loader = PathLoader(frames_path, start=self.start, stride=self.stride)
        path_iter = iter(path_loader)

        img_path, stop = next(path_iter)
        i = 0
        while not stop:
            if i >= self.steps:
                break
            self.states += [FrameState(i, img_path)]
            img_path, stop = next(path_iter)
            i += 1

    def _load_woko_frames(self, path):
        # Load PARKING specific frames
        frames_path = path + "images/"
        path_loader = PathLoader(frames_path, start=self.start, stride=self.stride)
        path_iter = iter(path_loader)

        img_path, stop = next(path_iter)
        i = 0
        while not stop:
            if i >= self.steps:
                break
            self.states += [FrameState(i, img_path)]
            img_path, stop = next(path_iter)
            i += 1

    def _load_kitti_poses(self, path):
        # Load KITTI specific poses
        poses_path = path + "poses/05.txt"
        poses = np.loadtxt(poses_path)
        self.poses = np.array([np.append(np.reshape(pose, (3, 4)), np.array([[0, 0, 0, 1]]), axis=0) for pose in poses], dtype=np.float32)


    def _load_malaga_poses(self, path):
        return None
    
    def _load_parking_poses(self, path):
        # Load PARKING specific poses
        poses_path = path + "poses.txt"
        poses = np.loadtxt(poses_path)
        self.poses = np.array([np.append(np.reshape(pose, (3, 4)), np.array([[0, 0, 0, 1]]), axis=0) for pose in poses], dtype=np.float32)

    def _load_woko_poses(self, path):
        return None

    def _malaga_file_filter(self, filename):
        return '_left.jpg' in filename
    
    def get_data(self):
        return self.k, self.poses, self.states

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Initialize DataLoader with a dataset
    loader = DataLoader(Dataset.KITTI)

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

    num_frames = len(loader)
    print(f"Number of frames: {num_frames}")

    # Display the first image
    plt.imshow(images[0], cmap='gray')
    plt.show()