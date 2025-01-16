from vis_nav_game import Player, Action
import pygame
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
from joblib import load
import re
import os
from maze_solver.maze_solver_Path_generate import maze_solver
from matplotlib.pyplot import imsave, imshow, figure, show, waitforbuttonpress
from time import strftime
import pickle
import matplotlib.pyplot as plt
from VO.VO_parl import VisualOdometry
import json


class KeyboardPlayer(Player):
    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.feature_extractor = None
        self.stored_features = None
        self.image_paths = None
        self.goal = None
        self.goal_loc=None
        self.curr=None
        self.loc_cod=None
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.maze_img=cv2.imread('maze_solver/Maze1.jpg',cv2.IMREAD_GRAYSCALE)
        self.FEATURES_PATH = "maze1features_joblib/image_features.joblib"
        self.PATHS_PATH = "maze1features_joblib/image_paths.joblib"
        self.image_location = "clean_images_final_maze1/"
        self.auto=False
        # self.path = np.array([[0,0],[1,0], [2,0], [3,0], [4,0],[5,0],[6,0],[6,1],[6,2],[7,2],[8,2]],dtype=float) #,[5,11],[2,11],[2,10],[4,10],[4,9],[2,9],[2,8],[3,8],[3,7],[2,7],[2,5],[1,5],[1,8],[0,8],[0,11],[-3,11],[-3,7],[-4,7],[-4,6],[-5,6],[-5,5],[-4,5],[-4,3],[-5,3],[-5,0],[-4,0],[-4,1],[-3,1],[-3,0],[0,0]],dtype=float)
        # self.path = np.array([[0,0],[0,0],[0,0],[0,0],[3,0], [3,4], [3,4],[3,4],[1,4],[1,4], [1,8],[0,8],[0,11],[-3,11],[-3,7],[-4,7],[-4,6],[-5,6],[-5,5],[-4,5],[-4,3],[-5,3],[-5,0],[-4,0],[-4,1],[-3,1],[-3,0],[0,0]],dtype=float)
        # # self.path = np.load("C:/Users/heman/OneDrive/Studies/SEM3/Perception/Project/vis_nav_player/solution_path.npy").astype(float)
        self.path = None
        self.head=np.array([[0], [1]])
        self.prev_head=None
        self.path_idx=0
        self.prev_fpv=None
        self.vo=VisualOdometry()
        self.angle=0
        self.dist=0
        self.trvl_dist=None
        self.b_im=None
        self.angle_thres=90
        self.dis_buff=0
        self.tmp_trgt=None
        self.chk_col=None
        self.hit_wal=False
        self.maze_cood_file = "maze1_img_cood.json"
        super(KeyboardPlayer, self).__init__()
        try:
            base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            self.feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
            self.stored_features = load(self.FEATURES_PATH)
            self.image_paths = load(self.PATHS_PATH)
            with open(self.maze_cood_file, 'r') as f:
                self.loc_cod = json.load(f)
            self.nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.stored_features)
            print("Feature matching system initialized successfully")
        except Exception as e:
            print(f"Error initializing feature matching system: {str(e)}")
        

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            # pygame.K_ESCAPE: Action.QUIT
        }
        
        # Load features
    @staticmethod
    def binary(image1):
        image=image1.copy()
        lo=np.array([198,180,150])
        hi=np.array([235,200,175])
        mask=cv2.inRange(image,lo,hi)
        image[mask!=0]=(235,235,235)
        lo=np.array([230,230,230])
        hi=np.array([249,249,249])
        mask=cv2.inRange(image,lo,hi)
        image[mask==0]=(0,0,0)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,image = cv2.threshold(image,12,255,cv2.THRESH_BINARY)
        kernel = np.ones((2,2),np.uint8) #Setting kernel for morphing
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel,iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,iterations=5) #Close morphing (Dilation followed by erosion eliminates black dots)
        image=(image//255).astype(np.uint8)
        return image

    def vo_reset(self):
        # print(f"Dist = {self.dist} Angle = {self.angle}")
        self.angle=0
        self.dist=0
        self.trvl_dist=None
        self.hit_wal=False
        # print("VO Reset!!")
    def get_vo(self):
        act=str(self.last_act).split(".")[1]
        if act=="IDLE": #or (self.prev_fpv-self.fpv).mean()<10
            return
        else:
            try:
                del_a, del_t= self.vo.get_pose(self.prev_fpv,self.fpv,act)
                self.dist = self.dist + del_t
                self.angle = self.angle + del_a
                # print(del_t)
            except:
                return
    def get_dir(self,p2,p1):
        direction=p2-p1
        dist=direction.mean()
        if dist>=0.5:
            dist=(abs(2*dist)//1)
        else:
            dist=(abs(2*dist))
        direction=direction//dist
        #Check condition for turning
        if (direction@self.head).mean()==0.0:
            if self.head[1]==0:
                check=direction.mean()
            else:
                check=direction.mean()*self.head[1]
            if self.head[0]==0:
                pass
            else:
                check=check*self.head[0]*-1
        else:
            check=None
        return check , direction , dist

    def extract_features(self, image_array):
        # Convert to PIL Image and resize
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        image = image.resize((224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        features = self.feature_extractor.predict(image_array, verbose=0)
        return features.flatten().astype(np.float32)
    
    def verify_image_match(self, im1, im2, threshold=0.85):
        """
        Verify image match using SIFT features and ratio test.
        
        Args:
            im1 (numpy.ndarray): First input image
            im2 (numpy.ndarray): Second input image
            threshold (float): Ratio test threshold for match filtering
        
        Returns:
            dict: Matching information including number of good matches and match ratio
        """
        try:
            # Convert images to grayscale if they're not already
            if len(im1.shape) == 3:
                gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = im1

            if len(im2.shape) == 3:
                gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = im2

            # Detect keypoints and compute descriptors
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            kp2, des2 = self.sift.detectAndCompute(gray2, None)

            # Check if descriptors were found
            if des1 is None or des2 is None:
                return {
                    'match_count': 0, 
                    'total_keypoints': (len(kp1), len(kp2)),
                    'match_ratio': 0.0,
                    'error': 'No descriptors found'
                }

            # Use kNN matching
            matches = self.flann.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < threshold * n.distance:
                    good_matches.append(m)

            # Calculate match ratio
            match_ratio = len(good_matches) / min(len(kp1), len(kp2)) if min(len(kp1), len(kp2)) > 0 else 0

            #print("\nDetailed Match Information:")
            #print(f"Total Keypoints - Image 1: {len(kp1)}, Image 2: {len(kp2)}")
            #print(f"Number of Good Matches: {len(good_matches)}")
            #print(f"Match Ratio: {match_ratio:.2%}")
            
            # Print specific match details
            #print("\nMatch Details:")
            #for i, match in enumerate(good_matches[:10], 1):
                #print(f"Match {i}:")
                #print(f"  Query Keypoint Index: {match.queryIdx}")
                #print(f"  Train Keypoint Index: {match.trainIdx}")
                #print(f"  Distance: {match.distance}")


            # Optional: Visualize matches (can be commented out if not needed)
            if len(good_matches) > 0:
                match_img = cv2.drawMatches(
                    gray1, kp1, 
                    gray2, kp2, 
                    good_matches[:min(len(good_matches), 50)], 
                    None, 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                #cv2.imshow('Image Matches', match_img)
                cv2.waitKey(1)

            return {
                'match_count': len(good_matches),
                'total_keypoints': (len(kp1), len(kp2)),
                'match_ratio': match_ratio,
                'matches': good_matches
            }

        except Exception as e:
            print(f"Error in image matching: {str(e)}")
            return {
                'match_count': 0, 
                'total_keypoints': (0, 0),
                'match_ratio': 0.0,
                'error': str(e)
            }

    def find_closest_match(self, target_features):
        distances, indices = self.nn_model.kneighbors([target_features])
        return indices[0][0]

    def extract_image_number(self, path):
        match = re.search(r'image_(\d+)', path)
        if match:
            return int(match.group(1))
        return None

    def find_matching_image(self, img):
        if img is None:
            print("No current view available")
            return

        try:
            # Find the top 5 closest feature matches
            target_features = self.extract_features(img)
            distances, indices = self.nn_model.kneighbors([target_features], n_neighbors=5)
            
            # Variables to track the best match
            best_match_index = None
            best_match_ratio = 0
            best_match_details = None

            # Iterate through top matches and verify
            for idx in indices[0]:
                matching_path = self.image_location + self.image_paths[idx]
                matching_image = cv2.imread(matching_path)
                
                # Verify the match using SIFT
                match_result = self.verify_image_match(img, matching_image)
                
                # Update best match if current match is better
                if match_result['match_ratio'] > best_match_ratio:
                    best_match_ratio = match_result['match_ratio']
                    best_match_index = idx
                    best_match_details = match_result

            # If we found a good match
            if best_match_index is not None:
                best_matching_path = self.image_location + self.image_paths[best_match_index]
                image_number = self.extract_image_number(best_matching_path)
                
                #print("\n--- Best Match Found ---")
                #print(f"Best Match Path: {best_matching_path}")
                #print(f"Best Match Ratio: {best_match_ratio:.2%}")
                #print(f"Match Count: {best_match_details['match_count']}")
                
                # Update goal and current image
                if self.goal is None and image_number is not None:
                    self.goal = image_number
                    print(f'Target image id: {self.goal}')
                else:
                    if image_number is not None:
                        self.curr = image_number
                        print(f"Current: {image_number} Target: {self.goal}")
                
                # Show the best matching image
                best_matching_image = cv2.imread(best_matching_path)
                if best_matching_image is not None:
                    cv2.imshow('Best Matching Image', best_matching_image)
                    cv2.waitKey(1)
            else:
                print("No suitable match found.")

        except Exception as e:
            print(f"Error in image matching: {str(e)}")

    def act(self):
        if ((self.auto == False)) :
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.last_act = Action.QUIT
                    return Action.QUIT
                if event.type == pygame.KEYDOWN:
                    if event.key==97:
                        self.auto= not self.auto
                        
                    elif event.key in self.keymap:
                        self.last_act |= self.keymap[event.key]
                    elif event.key == pygame.K_q:  #q for query
                        self.find_matching_image(self.fpv)

                    elif event.key==114:
                        self.find_matching_image(self.fpv)
                        maze_solver_obj = maze_solver(src=(int(self.loc_cod[str(self.curr)][0]),int(self.loc_cod[str(self.curr)][1])), dst=self.goal_loc, img=self.maze_img)
                        p = maze_solver_obj.find_shortest_path()
                        maze_solver_obj.drawPath(path=p, thickness=2)
                        cv2.imshow('Maze_Solve',maze_solver_obj.img)
                        cv2.waitKey(1)
                    elif event.key==98:
                        tmp=self.binary(self.fpv.copy())
                        plt.imshow(tmp,"gray")
                        plt.show()              
                    elif event.key==116:
                        print(f"Dist = {self.dist}\nAngle = {self.angle}\ntrvl_dist {self.trvl_dist}")
                    elif event.key==115:
                        self.vo_reset()
                    else:
                        # 
                        print(event.key)
                        self.show_target_images()
                if event.type == pygame.KEYUP:
                    if event.key in self.keymap:
                        self.last_act ^= self.keymap[event.key]
            return self.last_act
        elif ((pygame.event.peek(pygame.KEYDOWN) == False) and self.auto==True and (self.path is not None) and (self.prev_fpv is not None)):
            if len(self.path)==self.path_idx+1:
                self.auto=False
                print("Target Reached")
                return Action.IDLE
            self.get_vo()
            self.last_act=Action.IDLE

            check, direction, dist = self.get_dir(self.path[self.path_idx+1],self.path[self.path_idx])
            # #Check condition for turning

            if check is not None:
                if check>0:
                    if self.angle<self.angle_thres:
                        self.last_act=Action.RIGHT
                    else:
                        self.last_act=Action.IDLE
                        self.prev_head = self.head
                        self.head=direction
                        # self.auto=False
                        self.angle_thres=89
                        self.vo_reset()
                else:
                    if self.angle<self.angle_thres:
                        self.last_act=Action.LEFT
                    else:
                        self.last_act=Action.IDLE
                        self.prev_head = self.head
                        self.head=direction
                        # self.auto=False
                        self.angle_thres=89
                        self.vo_reset()
            else:
                if self.trvl_dist is None:
                    if dist>=1:
                        self.trvl_dist = (dist*6.2)#+((dist-1)*1)
                        # if dist>3:
                        #     self.trvl_dist = self.trvl_dist + dist*1.1
                    elif dist>0:
                        self.trvl_dist = 3
                    else:
                        self.trvl_dist = 0
                
                self.b_im=self.binary(self.fpv.copy())
                d_min = min(sum(self.b_im[:,120:200]))
                d_140_180=np.mean(sum(self.b_im[:,140:180]))
                if (self.path[self.path_idx+1] == self.tmp_trgt).all():
                    self.dist=self.dist+self.dis_buff
                    self.tmp_trgt=None
                    # print(f"dist {self.dist}")
                    # print(f"Trvl_dist {self.trvl_dist}")
                if ((self.dist<self.trvl_dist) and (((self.trvl_dist-self.dist)>4) or (d_140_180>12))):
                    if ((abs(abs(self.trvl_dist - self.dist)) <=1.1) and (len(self.path)>self.path_idx+2)): #
                        check, _, _ = self.get_dir(self.path[self.path_idx+2],self.path[self.path_idx+1])
                        if check is not None:
                            if check>0:
                                self.chk_col=314
                                # d_bim = sum(sum(self.b_im[:,314:319]))//6
                            else:
                                # d_bim = sum(sum(self.b_im[:,0:5]))//6
                                self.chk_col=0
                            d_bim = sum(sum(self.b_im[:,self.chk_col:self.chk_col+5]))//6
                            if d_bim>25:
                                # print(f"dist {self.dist} d_bm {d_bim}")
                                self.trvl_dist=self.dist+3
                            else:
                                self.trvl_dist+=1.1
                                # print("trvl dist increased")
                    elif ((d_min <=25) and (abs(self.trvl_dist - self.dist) >=5.5) and not (self.prev_head is None)):
                        self.last_act=Action.IDLE
                        d_l = np.mean(sum(self.b_im[:,0:160]))
                        d_r = np.mean(sum(self.b_im[:,160:319]))
                        # print(f"Cur Head {self.head}\nPrev_Head {self.prev_head}\nDire {d_r} {d_l}")
                        # print(f"D_min {d_min}\n Cur_Dis {self.dist}\ntrvl_dist {self.trvl_dist}")
                        # self.auto=False
                        delta = np.abs(self.prev_head)*0.5*((int(d_r<d_l)*2)-1)*np.sign(self.head.mean())
                        if self.prev_head[0]==0:
                            delta = delta*-1
                        # print(f"Delta {delta}\nPrev Head {self.prev_head}")
                        # if self.path_idx==0:    
                        #     new_loc = (delta.T)
                        # else:
                        new_loc = self.path[self.path_idx]+(delta.T)
                        # print(f"new_loc {new_loc}")
                        self.tmp_trgt = self.path[self.path_idx+1]
                        # if self.dis_buff == 0:    
                        #     
                        self.dis_buff+=self.dist
                        self.path = np.insert(self.path,self.path_idx,np.asarray(new_loc),axis=0)
                        # print(f"Path {self.path}\n\nLoc {self.path[self.path_idx]}\ntmp_trgt {self.tmp_trgt}")
                        self.vo_reset()
                        self.trvl_dist=None
                        return self.last_act
                    self.last_act=Action.FORWARD
                else:
                    # if dist==1:
                    # print(d_140_180)
                    d_140_180=np.mean(sum(self.b_im[:,130:190]))
                    d_140_180_max=np.max(sum(self.b_im[:,130:190]))
                    if (((d_140_180<35) and d_140_180_max<40) or (self.hit_wal==True)):
                        self.hit_wal=True
                        d_140 = sum(self.b_im[:,130])
                        d_180 = sum(self.b_im[:,190])
                        if abs(d_140-d_180)>=3:
                            if d_140>d_180: #Turn Right
                                self.last_act=Action.RIGHT
                            else:
                                self.last_act=Action.LEFT
                            return self.last_act
                        else:
                            # self.last_act=Action.IDLE
                            # print("Alligned")
                            self.hit_wal=False
                        # print(f"Hit Wall {self.hit_wal}")
                    self.vo_reset()
                    self.path_idx=self.path_idx+1
                    print(f"loc {self.path[self.path_idx]}")
                    if ((self.tmp_trgt is None) or (self.path[self.path_idx] == self.tmp_trgt).all()):
                        # print("Dist Buffer RESET")
                        self.dis_buff=0
                        self.tmp_trgt=None
                    self.trvl_dist=None
                    self.b_im=None
                    # self.auto=False
                return self.last_act
        else:
            if ((self.path is not None) and (self.prev_fpv is not None)):
                self.auto=False
            self.last_act=Action.IDLE
        return self.last_act


    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        match_dist=[]


        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        # Views names
        views = ["Front View", "Left View", "Back View", "Right View"]
        
        # Best match tracking
        best_overall_match = {
            'match_ratio': 0,
            'image_id': None,
            'best_view_index': 0,
        }

        # Calculate matches for each target view
        matches = []
        match_ratios = []
        
        for target_index, target in enumerate(targets):
            try:
                # Find top matches
                target_features = self.extract_features(target)
                distances, indices = self.nn_model.kneighbors([target_features], n_neighbors=5)
                
                # Variables to track best match for this view
                best_match_ratio = 0
                best_match_id = None
                
                # Verify matches
                for idx in indices[0]:
                    matching_path = self.image_location + self.image_paths[idx]
                    matching_image = cv2.imread(matching_path)
                    
                    # Verify the match using SIFT
                    match_result = self.verify_image_match(target, matching_image)
                    
                    # Update best match if current match is better
                    if match_result['match_ratio'] > best_match_ratio:
                        best_match_ratio = match_result['match_ratio']
                        best_match_id = self.extract_image_number(matching_path)
                
                # Store match information
                matches.append(best_match_id)
                match_ratios.append(best_match_ratio)
                
                # Update best overall match
                if best_match_ratio > best_overall_match['match_ratio']:
                    best_overall_match['match_ratio'] = best_match_ratio
                    best_overall_match['image_id'] = best_match_id
                    best_overall_match['best_view_index'] = target_index
                    # best_overall_match['best_idx'] = idx
            
            except Exception as e:
                print(f"Error matching view {target_index}: {str(e)}")
                matches.append(None)
                match_ratios.append(0)

        # Set the goal to the best matched image
        if best_overall_match['image_id'] is not None:
            self.goal = best_overall_match['image_id']
            cv2.imshow("Best Macthing Target Image" , cv2.imread(self.image_location + f"image_{str(self.goal).zfill(4)}.png"))
            cv2.waitKey(1)
            #print(f"Best Overall Target image id: {self.goal}")
            #print(f"Best Match View Index: {best_overall_match['best_view_index']}")
            #print(f"Best Match Ratio: {best_overall_match['match_ratio']:.2%}")

        if self.goal is not None:
            print(f"Self.GOAL {self.goal}")
            start_point = (150,255)
            self.goal_loc = (int(self.loc_cod[str(self.goal)][0]),int(self.loc_cod[str(self.goal)][1])) #(30,24) # TBD Based on indexing code
            # end_point=(30,24) (114,55) (255,81)
            # self.goal_loc=(255,81)
            print(f"Goal Coordinate : {self.goal_loc}")
            maze_solver_obj = maze_solver(src=start_point, dst=self.goal_loc, img=self.maze_img)
            p = maze_solver_obj.find_shortest_path()
            self.path = maze_solver_obj.gen_path(p)
            # print(self.path)
            maze_solver_obj.drawPath(path=p, thickness=2)
            cv2.imshow('Maze_Solve',maze_solver_obj.img)
            cv2.waitKey(1)
        # Add text overlays for each quadrant
        positions = [(h_offset, w_offset),
                    (int(h / 2) + h_offset, w_offset),
                    (h_offset, int(w / 2) + w_offset),
                    (int(h / 2) + h_offset, int(w / 2) + w_offset)]

        for i, pos in enumerate(positions):
            # Prepare match text with image ID and match ratio
            match_text = f'{views[i]}: {matches[i] if matches[i] is not None else "No Match"}'
            ratio_text = f'Ratio: {match_ratios[i]:.2%}' if matches[i] is not None else ''
            
            # Highlight the best matching view
            text_color = (0, 0, 255) if i == best_overall_match['best_view_index'] else color
            
            # Draw view name and image ID
            cv2.putText(concat_img, match_text, pos, font, size, text_color, stroke, line)
            
            # Draw match ratio below the view name
            cv2.putText(concat_img, ratio_text, 
                        (pos[0], pos[1] + 25),  # Offset vertically
                        font, size, text_color, stroke, line)

        # Display and save the final image
        cv2.imshow('KeyboardPlayer:target_images', concat_img)
        cv2.imwrite('target_with_matches.jpg', concat_img)
        cv2.waitKey(1)


    def set_target_images(self, images):
        super(KeyboardPlayer, self).set_target_images(images)
        self.show_target_images()

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    def pre_navigation(self) -> None:
        pass

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.prev_fpv=self.fpv
        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1] 
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()
        # if self.prev_fpv is not None:
        #     self.get_vo()



if __name__ == "__main__":
    import logging
    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    import vis_nav_game as vng
    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')
    vng.play(the_player=KeyboardPlayer())