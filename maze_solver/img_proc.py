# import cv2
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load and convert the image to grayscale
image_path = Path('img/maze_1.jpeg').absolute().__str__()
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Thresholding to get a binary image
# Invert colors for skeletonization
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# plt.imshow(binary_image)
# plt.show()

dilatation_size = 2
element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
dilated_edges = cv2.dilate(binary_image, element)

# plt.imshow(dilated_edges)
# plt.show()

# Find contours and get bounding box of the largest contour (the maze boundary)
contours, _ = cv2.findContours(
    dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
diff = 2
# Crop the image to the bounding box of the mazezx
cropped_image = dilated_edges[y+diff:y+h-diff, x+diff:x+w-diff]
# cropped_image = dilated_edges[y:y+h, x:x+w]
plt.imshow(cropped_image)
plt.show()

# Step 3: Define the desired grid size
# Adjust based on the number of rows and columns you want
grid_size = (30, 30)
cell_height = cropped_image.shape[0] // grid_size[0]
cell_width = cropped_image.shape[1] // grid_size[1]

# Initialize the maze matrix
maze_matrix = np.zeros(grid_size, dtype=int)

# Step 4: Analyze each cell individually
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        # Extract the cell region from the cropped binary image
        cell = cropped_image[i * cell_height:(i + 1) *
                             cell_height, j * cell_width:(j + 1) * cell_width]

        # Calculate the proportion of wall vs path in the cell
        wall_pixels = np.sum(cell == 0)  # Count of wall pixels
        path_pixels = np.sum(cell == 255)  # Count of path pixels

        # Determine if the cell is primarily a wall or a path
        if wall_pixels > path_pixels:
            maze_matrix[i, j] = -1  # Wall
        else:
            maze_matrix[i, j] = 0  # Path

# Display the final matrix for visual confirmation
plt.imshow(maze_matrix, cmap='gray')
plt.title("Maze Grid Matrix")
plt.show()

# Print the matrix
print(maze_matrix)
