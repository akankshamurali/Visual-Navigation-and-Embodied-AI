import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import concurrent.futures

class VisualOdometry():
    def __init__(self):
        self.K = self._load_calib()
        self.sift = cv2.SIFT_create(500)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    @staticmethod
    def _load_calib():
        K = np.array([[92, 0, 160],
                      [0, 92, 120],
                      [0, 0, 1]])
        return K

    def detect_and_compute(self, img):
        return self.sift.detectAndCompute(img, None)

    def get_matches(self, im1, im2):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self.detect_and_compute, im1)
            future2 = executor.submit(self.detect_and_compute, im2)
            kp1, des1 = future1.result()
            kp2, des2 = future2.result()

        matches = self.flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = [m for m, n in matches if m.distance < 0.5 * n.distance]

        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1, q2

    def get_pose(self, im1, im2, act):
        q1, q2 = self.get_matches(im1, im2)
        
        try:
            E, mask1 = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask2 = cv2.recoverPose(E, q1, q2, self.K)
        except:
            R = np.eye(3)
            t = np.zeros((3,1))

        if act in ["RIGHT", "LEFT"]:
            t = np.zeros_like(t)
        elif act in ["FORWARD", "BACKWARD"]:
            R = np.eye(3)
            # t[0:1] = 0

        rot_vec, _ = cv2.Rodrigues(R)
        angle = np.degrees(np.linalg.norm(rot_vec))
        if angle > 5:
            angle = 2.5

        return angle, -t[-1][0]