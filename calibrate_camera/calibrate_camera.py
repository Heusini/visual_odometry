import glob
import sys

import cv2 as cv
import numpy as np


def find_checker_board(images, square_size=1):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * square_size
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (6, 8), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (6, 8), corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(500)
        else:
            cv.imshow("gray", gray)
            cv.waitKey(500)
        # print(rvecs)
        # print(tvecs)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("Out:")
    print(ret)
    print(mtx)
    print(dist)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print("New:")
    print(newcameramtx)
    print(roi)
    cv.destroyAllWindows()


if __name__ == "__main__":
    path = sys.argv[1]
    images = glob.glob(f"{path}/*.png")
    find_checker_board(images, square_size=30)
