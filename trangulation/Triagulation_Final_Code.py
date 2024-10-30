import cv2 as cv
import numpy as np

# Load the images
left_image = cv.imread('left.png')
right_image = cv.imread('right.png')

# Rotate the right image by 180 degrees
rotated_image = cv.flip(right_image, -1)
cv.imwrite('right.png', rotated_image)

# Convert to grayscale for feature matching
img1 = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(rotated_image, cv.COLOR_BGR2GRAY)

import numpy as np

# Rotation matrix (R) for cam0
R = np.array([
    [0.9999313912417018, -0.0023139054373197965, 0.011482972222461762],
    [0.002353841678837691, 0.9999912245858043, -0.003465570766066675],
    [-0.011474852451585301, 0.0034923620961737592, 0.9999280629966356]
])

# Translation vector (T) for cam0
T = np.array([
    [-0.9998],
    [0.00145],
    [0.0455]
])

# Camera intrinsic matrix (mtx) for cam0
mtx = np.array([
    [555.6627242364661, 0, 342.5725306057865],
    [0, 555.8306341927942, 215.26831427862848],
    [0, 0, 1]
])

# Projection matrices
P1 = np.hstack((mtx, np.zeros((3, 1))))  # Projection matrix for the left camera
P2 = mtx @ np.hstack((R, T.reshape(-1, 1)))  # Projection matrix for the right camera

print("Projection matrix for the left camera (P1):")
print(P1)

# Initialize SIFT detector
sift = cv.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher to find matches between descriptors
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test to keep only good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        good_matches.append(m)

# Extract the matched points
pts1 = np.float64([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2).T
pts2 = np.float64([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2).T

# Triangulate points to get 4D homogeneous coordinates
points4D = cv.triangulatePoints(P1, P2, pts1, pts2)

# Convert homogeneous coordinates to 3D
points3D = points4D[:3] / points4D[3]

# Transpose the points for proper formatting
pts1_transposed = pts1.T  # Shape: (n, 2) for 2D points
points3D_transposed = points3D.T  # Shape: (n, 3) for 3D points

# Extract first 20 points for writing (or adjust as needed)
points3D_20 = points3D_transposed[:20]  # First 20 3D points (U)
pts1_20 = pts1_transposed[:20]  # First 20 2D points (u)

# Write the formatted 3D points (U) and 2D points (u) to a text file
with open('points_output.txt', 'w') as f:
    f.write("U = [\n")
    for i in range(points3D_20.shape[0]):
        f.write(f"  {points3D_20[i, 0]:10.4f} {points3D_20[i, 1]:10.4f} {points3D_20[i, 2]:10.4f}\n")
        print(f"Point {i+1}: {points3D_20[i]}")
    f.write("]; %noisefree 3d points\n\n")
    
    f.write("u = [\n")
    for i in range(pts1_20.shape[0]):
        f.write(f"  {pts1_20[i, 0]:10.4f} {pts1_20[i, 1]:10.4f}\n")
        print(f"Point {i+1}: {pts1_20[i]}")
    f.write("]; % noisy image observations\n")

print("Points saved to 'points_output.txt'.")
