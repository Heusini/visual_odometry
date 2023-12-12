import os
import numpy as np
import cv2
import vo_pipeline as vo

def main():
    DATAPATH = 'data/kitti/05/image_0/'
    I_0 = 0
    I_1 = 1    

    imgs = vo.helpers.load_images(DATAPATH)

    print(imgs.shape)
    imgs_subarray = imgs[I_0:I_1]

    frame_state = vo.initialization(imgs_subarray, I_1)

    # continuous operation
    for i in range(I_1 + 1, imgs.shape[0]):
        frame_state = vo.process_frame(frame_state, imgs[i-1], imgs[i])


if __name__ == '__main__':
    main()