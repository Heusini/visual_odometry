import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_all(states):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    def update(t):
        ax1.clear()
        ax1.imshow(states[t].img, cmap='gray')
        ax1.scatter(states[t].landmarks[:, 0], states[t].landmarks[:, 1], color='r')
        ax1.set_title("Current Image")

        ax2.clear()
        ax2.plot([state.landmarks.shape[0] for state in states[max(0, t-20):t+1]])
        ax2.set_title("tracked landmarks over last 20 frames.")
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Number of landmarks')

        ax3.clear()
        ax3.plot([state.cam_to_world[0, 3] for state in states[:t+1]], [state.cam_to_world[1, 3] for state in states[:t+1]])
        ax3.set_title("Full Trajectory")

        ax4.clear()
        ax4.plot([state.cam_to_world[0, 3] for state in states[max(0, t-20):t+1]], [state.cam_to_world[1, 3] for state in states[max(0, t-20):t+1]])
        ax4.scatter(states[t].landmarks[:, 0], states[t].landmarks[:, 1], color='r')
        ax4.set_title("trajectory of last 20 frames and landmarks")

    anim = FuncAnimation(fig, update, repeat=False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from ..tracking import TrackManager
    from dataloader import DataLoader, Dataset
    from ..initialization import initialize
    from ..plot_points_cameras import plot_points_cameras
    from ..twoDtwoD import FeatureDetector
    from ..pnp import pnp
    import numpy as np
    import cv2 as cv

    # select dataset
    dataset = Dataset.KITTI

    # load data
    # if you remove steps then all the images are used
    loader = DataLoader(dataset, start=105, stride=1, steps=90)
    print("Loading data...")

    loader.load_data()
    print("Data loaded!")

    K, poses, states = loader.get_data()
    print("Data retrieved!")

    init_frame_indices = DataLoader.best_params[dataset.name]["init_frame_indices"]
    initialize(states[init_frame_indices[0]], states[init_frame_indices[1]], K)
    init_states = [states[init_frame_indices[0]], states[init_frame_indices[1]]]

    track_manager = TrackManager(
        angle_threshold=2.5*np.pi/180,
        same_keypoints_threshold=0.9,
        max_track_length=10,
    )

    for t in range(60):
        states[t+1] = pnp(states[t], states[t+1], K)

        track_manager.start_new_track(
            states[t],
            check_keypoints=True)

        track_manager.update(
            t,
            img_i=cv.imread(states[t].img_path),
            img_j=cv.imread(states[t+1].img_path))

        landmarks, keypoints = track_manager.get_new_landmarks(
            t+1,
            min_track_length=5,
            frame_states=states,
            K=K,
            compare_to_landmarks=True)

        states[t+1].landmarks = np.hstack([states[t+1].landmarks, landmarks])
        states[t+1].keypoints = np.hstack([states[t+1].keypoints, keypoints])

        plot_all(states)