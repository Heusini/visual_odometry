import cv2
import numpy as np
from visual_odometry.utils.geometry import calculate_fundamental_matrix, calculate_essential_matrix
from visual_odometry.utils.image_processing import detect_features, match_features


class Initialization:

    def __init__(self):
        self.points1 = None
        self.points2 = None

    def initialize(self, frame1, frame2, K):

        keypoints1, descriptors1 = detect_features(frame1)
        keypoints2, descriptors2 = detect_features(frame2)

        matches = match_features(descriptors1, descriptors2)

        # # Track these features to the second frame using KLT
        # keypoints2, _ = cv2.calcOpticalFlowPyrLK(frame1, frame2, keypoints1, None)

        self.points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        self.points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        print("points1.shape:", self.points1.shape)
        print("points2.shape:", self.points2.shape)
        print("points1 dtype: ", self.points1.dtype)
        print("points2 dtype: ", self.points2.dtype)

        E, mask_e = calculate_essential_matrix(self.points1, self.points2, K)

        inliers = mask_e.ravel() == 1

        _, R, t, _ = cv2.recoverPose(E, self.points1[inliers], self.points2[inliers], K)

        points_homogeneous = cv2.triangulatePoints(np.eye(3, 4), np.hstack((R, t)), self.points1[inliers].T, self.points2[inliers].T)
        landmarks = cv2.convertPointsFromHomogeneous(points_homogeneous.T)
        print("landmarks shape: ", landmarks.shape)

        pose = np.hstack((R.T, -R.T @ t))

        return pose, landmarks

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import plotly.graph_objects as go
    from visual_odometry.utils.dataloader import DataLoader, Dataset

    # Initialize DataLoader with a dataset
    loader = DataLoader(Dataset.KITTI)
    print("Loading data...")

    loader.load_data()
    print("Data loaded!")

    K, poses, images = loader.get_data()
    print("Data retrieved!")

    frame1_gray = images[0]
    frame2_gray = images[2]

    init = Initialization()

    pose, landmarks = init.initialize(frame1_gray, frame2_gray, K)


    # dynamic 3D plot
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=init.points1[:, 0], y=init.points1[:, 1], z=np.zeros(init.points1.shape[0]),
                               mode='markers', marker=dict(size=3, color='red'), name='Frame 1 keypoints'))

    fig.add_trace(go.Scatter3d(x=init.points2[:, 0], y=init.points2[:, 1], z=np.ones(init.points2.shape[0]),
                               mode='markers', marker=dict(size=3, color='blue'), name='Frame 2 keypoints'))

    for i in range(len(init.points1)):
        fig.add_trace(go.Scatter3d(x=[init.points1[i, 0], init.points2[i, 0]], y=[init.points1[i, 1], init.points2[i, 1]], z=[0, 1],
                                   mode='lines', line=dict(color='green'), showlegend=False))

    fig.add_trace(go.Scatter3d(x=[pose[0, 3]], y=[pose[1, 3]], z=[pose[2, 3]],
                               mode='markers', marker=dict(size=5, color='yellow'), name='Camera poses: ground truth'))

    fig.add_trace(go.Scatter3d(x=landmarks[:, 0, 0], y=landmarks[:, 0, 1], z=landmarks[:, 0, 2],
                               mode='markers', marker=dict(size=2, color='purple'), name='Landmarks'))

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Frame'),
                      title='Keypoint matches, camera poses and landmarks between two frames')

    fig.show()


    # 2D plot
    # fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    #
    # ax[0].imshow(frame1_gray, cmap='gray')
    # ax[0].scatter(init.points1[:, 0], init.points1[:, 1], color='red')
    # ax[0].set_title('Frame 1 with keypoints')
    #
    # ax[1].imshow(frame2_gray, cmap='gray')
    # ax[1].scatter(init.points2[:, 0], init.points2[:, 1], color='red')
    # ax[1].set_title('Frame 2 with keypoints')
    #
    # plt.show()

    # static 3D plot
    # plt.ion()
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(init.points1[:, 0], init.points1[:, 1], zs=0, zdir='z', s=20, depthshade=True, color='red')
    # ax.scatter(init.points2[:, 0], init.points2[:, 1], zs=1, zdir='z', s=20, depthshade=True, color='blue')
    #
    # for i in range(len(init.points1)):
    #     ax.plot([init.points1[i, 0], init.points2[i, 0]], [init.points1[i, 1], init.points2[i, 1]], zs=[0, 1], zdir='z', color='green')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Frame')
    # ax.set_title('Keypoint matches between two frames')
    #
    # plt.show()
