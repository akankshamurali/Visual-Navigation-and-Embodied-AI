import os
import numpy as np
import cv2
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
import json
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

    def get_matches(self,im1,im2):
        kp1, des1 = self.sift.detectAndCompute(im1, None)
        kp2, des2 = self.sift.detectAndCompute(im2, None)
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)
 
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1, q2

    def get_pose(self, im1, im2,act): 


        q1,q2=self.get_matches(im1, im2)
        # print(q1.shape)
        try:
            E, mask1 = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask2 = cv2.recoverPose(E, q1, q2, self.K)
        except:
            R=np.eye(3)
            t=np.zeros((3,1)) 
        # R,t = self.decomp_essential_mat(E,q1,q2)
        # print(t)
        # print(t.shape())
        if act=="RIGHT" or act=="LEFT":
            t=np.zeros_like(t)
        elif act =="FORWARD" or act=="BACKWARD":
            # if len(np.unique(im1))<3 or len(np.unique(im2))<3:
            #     img3=np.concatenate((im1, im2), axis=1)
            #     plt.imshow(img3)
            #     plt.show()
            R=np.eye(3)
            t[0:1]=0
        # else:
        #     R=np.eye(3)
        #     t=np.zeros_like(t)
        # transformation_matrix = self._form_transf(R, np.squeeze(t))
        rot_vec, _ = cv2.Rodrigues(R)
        angle=np.degrees(np.linalg.norm(rot_vec))
        if angle>5:
            angle=2.5
        # print(t[-1][0])
        # print(angle)
        return angle, -t[-1][0]


# def main():
#     data_dir = "D:\\Studies_2\\exploration_data\\images\\"  # Try KITTI_sequence_2 too
#     vo = VisualOdometry()
#     # gt_path = []
#     estimated_path = []
#     with open('D:\\Studies_2\\exploration_data\\output.json', 'r') as file:
#         act_data = json.load(file)
#     for i in range(150):
#         if i == 0:
#             cur_pose = np.eye(4)
#         else:
#             prev_act=act_data[i-1]
#             curr_act=act_data[i]
#             curr_image=curr_act["image"]
#             prev_image=prev_act["image"]
#             if len(prev_act["action"])==1:    
#                 act=prev_act["action"][0]
#             else:
#                 act="BOTH"
#             img1=cv2.imread(data_dir+prev_image,cv2.IMREAD_GRAYSCALE)
#             img2=cv2.imread(data_dir+curr_image,cv2.IMREAD_GRAYSCALE)
#             q1, q2 = vo.get_matches(img1,img2)
#             transf = vo.get_pose(q1, q2,act)

#             prev_pose=cur_pose
#             cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
#             # if act=="RIGHT" or act=="LEFT": 
#             #     print(act) 
#             #     print(cur_pose)
#             #     print(" ")
#             # gt_path.append([0, 0])
#             del_x=(cur_pose[0,3]-prev_pose[0,3])
#             del_y=(cur_pose[2,3]-prev_pose[2,3])
#             dist=np.sqrt((del_x*del_x)+(del_y*del_y))
#             if abs(del_x)>abs(del_y):
#                 if del_x>0:  
#                     estimated_path.append((prev_pose[0, 3]+dist, prev_pose[2, 3]))
#                     cur_pose[0,3]=prev_pose[0, 3]+dist
#                     cur_pose[2,3]=prev_pose[2, 3]
#                 else:
#                     estimated_path.append((prev_pose[0, 3]-dist, prev_pose[2, 3]))
#                     cur_pose[0,3]=prev_pose[0, 3]-dist
#                     cur_pose[2,3]=prev_pose[2, 3]
#             elif abs(del_y)>abs(del_x):
#                 if del_y>0:    
#                     estimated_path.append((prev_pose[0, 3], prev_pose[2, 3]+dist))
#                     cur_pose[0,3]=prev_pose[0, 3]
#                     cur_pose[2,3]=prev_pose[2, 3]+dist
#                 else:
#                     estimated_path.append((prev_pose[0, 3], prev_pose[2, 3]-dist))
#                     cur_pose[0,3]=prev_pose[0, 3]
#                     cur_pose[2,3]=prev_pose[2, 3]-dist
#             else:
#                 estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
#             # Add keyframe
#             # kp, des = vo.sift.detectAndCompute(vo.images[i], None)
#             # vo.add_keyframe(cur_pose, img2, kp, des)

#             # # Detect loop closure
#             # vo.detect_loop_closure(cur_pose)

#             # # Optimize the map
#             # vo.optimize()
#     # print(curr_image)
#     # estimated_path_smooth = vo.smooth_path(estimated_path, 10)
#     # x_coords, y_coords = zip(*estimated_path_smooth)
#     # plt.scatter(x_coords, y_coords)
#     # plt.show()
#     x_coords, y_coords = zip(*estimated_path)
#     plt.scatter(x_coords, y_coords)
#     plt.show()
#     # plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")

# if __name__ == "__main__":
#     main()