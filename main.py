import os
import numpy as np
import cv2
import vo_pipeline as vo

def main():
    DATAPATH = 'data/kitti/05/image_0/'
    I_0 = 0
    I_1 = 1    

    imgs = []
    for filename in os.listdir(DATAPATH):
        if filename.endswith('.png'):
            img = cv2.imread(DATAPATH + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgs.append(img)

    imgs = np.array(imgs)

    print(imgs.shape)
    imgs_subarray = imgs[I_0:I_1]

    frame_state = vo.initialization(imgs_subarray)

    for i in range(I_1, imgs.shape[0]):
        print(i)
        # TODO: make sure indices here are right !
        frame_state = vo.process_frame(frame_state, imgs[i-1], imgs[i])


if __name__ == '__main__':
    main()