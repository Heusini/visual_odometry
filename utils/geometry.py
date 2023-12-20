import numpy as np
import cv2

def calculate_fundamental_matrix(points1, points2):
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.RANSAC, 1, 0.9999, 50000)
    return F, mask

def calculate_essential_matrix(points1, points2, K):
    E, mask_e = cv2.findEssentialMat(points1, points2, K, cv2.RANSAC, 0.9999, 1, 50000)
    return E, mask_e

def calculate_essential_matrix_from_fundamental_matrix(F, K):
    return K.T @ F @ K