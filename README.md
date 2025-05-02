This project implements an embodied AI agent capable of navigating a maze using visual input alone. The agent uses a camera feed to perceive its environment, extract semantic features, compute the optimal path, and autonomously navigate toward a goal. This combines Visual Odometry, deep visual feature matching, and path planning to simulate embodied intelligence.

Problem Statement
The task is to guide a robot through a maze and reach a specific target in the shortest possible time. The robot must rely only on its onboard camera, without any prior map or GPS. The system is built to:
Map visual data to spatial positions using Visual Odometry
Recognize target locations by comparing features of current and past views
Use a maze-solving algorithm (Dijkstra's) for optimal path planning
Navigate autonomously or via keyboard-based manual control

Methodology
Key Steps:
1.Data Cleaning: Cleaned image sequence from exploration phase
2.Manual Mapping: Visual frames manually aligned to a grid
3.Visual Odometry: Estimated position change from image deltas
4.Feature Extraction:
Used EfficientNetB0 to extract deep features
Compared with VGG16, ResNet50V2, and hybrid models
5.Feature Matching:
Used SIFT for keypoint validation
FLANN-based matcher for ratio test verification
k-NN used for identifying the most similar view
6.Maze Solver: Implemented Dijkstra's Algorithm for path planning
7.Binary Image Analysis:
Thresholded view used to identify obstacles
Morphology operations enhance clarity for safer movement
8.Navigation Execution:
act() method handles both manual and autonomous movement
Detects obstacles and corrects orientation during traversal

Outcomes
Successfully navigated Maze 1 and Maze 2 using both automated and manual methods
Real-time dynamic obstacle avoidance using binary thresholding
Visual recognition of the goal image using deep + classical matching
