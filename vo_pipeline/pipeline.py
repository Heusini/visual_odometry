import os
import numpy as np
import cv2
import vo_pipeline as vo

def pipeline():
    DATAPATH = 'data/kitti/05/image_0/'
    I_0 = 0
    I_1 = 1    

    K = np.array(
        [
            [7.188560000000e02, 0, 6.071928000000e02],
            [0, 7.188560000000e02, 1.852157000000e02],
            [0, 0, 1],
        ]
    )

    imgs = vo.helpers.load_images(DATAPATH)

    print(imgs.shape)
    imgs_subarray = imgs[I_0:I_1]

    (kp_a, kp_b), (des_a, des_b) = vo.correspondence(imgs_subarray)

    landmarks, camera_a, camera_b = vo.sfm(kp_a, kp_b, K)

    frame_state =  vo.FrameState(I_1, K, kp_b, des_b, landmarks, camera_b)

    # continuous operation
    for i in range(I_1 + 1, imgs.shape[0]):
        frame_state = vo.process_frame(frame_state, imgs[i])


if __name__ == '__main__':
    pipeline()