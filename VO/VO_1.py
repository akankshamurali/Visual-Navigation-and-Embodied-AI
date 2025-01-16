import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import json
class VisualOdometry():
    def __init__(self):

        # Load Calibration Matrix
        self.K= np.array([[92, 0, 160], 
                      [0, 92, 120],
                      [0, 0, 1]])

        # Initialize Feature extraction and Matching
        self.sift = cv2.SIFT_create(5000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    
    def get_matches(self,im1,im2): 
    # Find good matches between given images Im1=Previous Image Im2=Current Image
        kp1, des1 = self.sift.detectAndCompute(im1, None)
        kp2, des2 = self.sift.detectAndCompute(im2, None)
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance:
                good.append(m)

        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2


    def get_pose(self, im1, im2,act): 
    # Estimate change in camera pose based on given images
    # Im1=Previous_Image Im2=Current_Image
    # Returns Unsigned change in angle in degrees
    # Returns signed distance traveled 
        q1,q2=self.get_matches(im1, im2)
        try:
            E, mask1 = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.99, threshold=1.0)
            _, R, t, mask2 = cv2.recoverPose(E, q1, q2, self.K)
        except:
            R=np.eye(3)
            t=np.zeros((3,1))
        if act=="RIGHT" or act=="LEFT":
            t=np.zeros_like(t)
        elif act =="FORWARD" or act=="BACKWARD":
            R=np.eye(3)
            t[0:1]=0
        rot_vec, _ = cv2.Rodrigues(R)
        angle=np.degrees(np.linalg.norm(rot_vec))
        # Sometimes angle reading jumps to very large hence thresholding
        # 2.5 is most probable value of angle found by trial error
        if angle>5:
            angle=2.5
        return angle, -t[-1][0]
def main():

# Use cleaning file before running this
# Action Data format for 1st maze and Midterm maze is different, Change code accordingly 

    data_dir = "D:\\Studies_2\\exploration_data\\images\\"  # Exploration Images Path
    vo = VisualOdometry()
    with open('D:\\Studies_2\\exploration_data\\output.json', 'r') as file: # Load Action data for exploration
        act_data = json.load(file)

    curr_loc = [0,0] 
    '''
    curr_loc will update by 1 unit in heading direction after 
    sum of travel distance >= 12.5 then reset travel distance for next
    '''

    head = [0,1] 
    '''
    This is a binary variable which tells current heading direction
    Heading direction X,Y will update as per rotation direction after
    sum of rotation >= 89.5 deg then reset rotation for next
    '''
if __name__ == "__main__":
    main()