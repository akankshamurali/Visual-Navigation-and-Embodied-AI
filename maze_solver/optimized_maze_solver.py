
import cv2
import matplotlib.pyplot as plt
import numpy as np
import heapq
import time

class Vertex:
    def __init__(self, x_coord, y_coord):
        self.x = x_coord
        self.y = y_coord
        self.d = float('inf')
        self.parent = None
        self.processed = False

class MazeSolver:
    def __init__(self, src, dst, img=None, img_path=None, dilatation_size=2):
        if img is not None:
            self.img = self.proc_img(image=img, dilatation_size=dilatation_size)
        else:
            self.img = self.proc_img(image_path=img_path, dilatation_size=dilatation_size)
        self.src = src
        self.dst = dst

    def get_neighbors(self, mat, r, c):
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = [
            mat[r + dr][c + dc]
            for dr, dc in offsets
            if 0 <= r + dr < mat.shape[0] and 0 <= c + dc < mat.shape[1] and not mat[r + dr][c + dc].processed
        ]
        return neighbors

    def get_distance(self, img, u, v):
        return 0.1 + np.sum((img[v[0], v[1]] - img[u[0], u[1]])**2)

    def draw_path(self, path, thickness=1):
        cv2.circle(self.img, self.src, 2, (255, 0, 0), 8)
        cv2.circle(self.img, self.dst, 2, (0, 0, 255), 8)
        for i in range(len(path) - 1):
            cv2.line(self.img, tuple(path[i]), tuple(path[i+1]), (255, 0, 0), thickness)

    def find_shortest_path(self):
        src, dst = self.src, self.dst
        img = self.img
        pq = []
        heapq.heappush(pq, (0, src))
        rows, cols = img.shape[:2]
        matrix = np.array([[Vertex(c, r) for c in range(cols)] for r in range(rows)])
        matrix[src[1]][src[0]].d = 0

        while pq:
            dist, (x, y) = heapq.heappop(pq)
            current = matrix[y][x]
            if current.processed:
                continue
            current.processed = True
            if (x, y) == dst:
                break

            for neighbor in self.get_neighbors(matrix, y, x):
                new_dist = dist + self.get_distance(img, (y, x), (neighbor.y, neighbor.x))
                if new_dist < neighbor.d:
                    neighbor.d = new_dist
                    neighbor.parent = (x, y)
                    heapq.heappush(pq, (new_dist, (neighbor.x, neighbor.y)))

        path = []
        current = matrix[dst[1]][dst[0]]
        while current is not None:
            path.append((current.x, current.y))
            current = matrix[current.parent[1]][current.parent[0]] if current.parent else None
        return path[::-1]

    def proc_img(self, img_path=None, image=None, dilatation_size=2):
        if image is None:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1))
        dilated_edges = cv2.dilate(binary_image, element)

        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = dilated_edges[y:y+h, x:x+w]

        inv_cropped_img = cv2.bitwise_not(cropped_image)
        img = cv2.cvtColor(inv_cropped_img, cv2.COLOR_GRAY2BGR)
        return img

if __name__ == '__main__':
    st=time.time()
    img = cv2.imread('maze.jpg', cv2.IMREAD_GRAYSCALE)
    src = (150, 265)
    dst = (10, 10)
    solver = MazeSolver(src=src, dst=dst, img=img)
    path = solver.find_shortest_path()
    solver.draw_path(path, thickness=2)
    print(time.time()-st)
    plt.figure(figsize=(7, 7))
    plt.imshow(solver.img)
    plt.show()
