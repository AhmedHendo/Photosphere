import cv2
import numpy as np
import matplotlib.pyplot as plt

def photosphere (img1, img2):
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher()

    # Find the 2 best matches for each descriptor
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []

    for m, n in matches:
        good_matches.append(m)
            
    # ✅ Check for enough matches
    if len(matches) < 4:
        print("Not enough good matches found. Skipping this image.")
        return image1
            
    points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    # Convert 2D points to 3D homogeneous coordinates by adding the third coordinate as 1
    points1_homogeneous = np.hstack([points1, np.ones((points1.shape[0], 1))])  # Shape (n, 3)
    points2_homogeneous = np.hstack([points2, np.ones((points2.shape[0], 1))])  # Shape (n, 3)

    # Prepare the matrix A and the vector b for the homography calculation
    A = []
    b = []

    # Loop through each correspondence and build the system of equations
    for i in range(len(points1)):
        x1, y1, _ = points1_homogeneous[i]  # Homogeneous coordinates of points in image 1
        x2, y2, _ = points2_homogeneous[i]  # Homogeneous coordinates of points in image 2
        
        # Construct two rows of the matrix A
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1])
        
        # Construct the vector b
        b.append(x2)
        b.append(y2)

    # Convert A and b to numpy arrays
    A = np.array(A)
    b = np.array(b)

    # Solve the system A * h = b using least squares to compute the homography
    h = np.linalg.lstsq(A, b, rcond=None)[0]

    # Reshape h to form the 3x3 homography matrix
    h_matrix = np.array([[h[0], h[1], h[2]], 
                [h[3], h[4], h[5]], 
                [h[6], h[7], 1]])

    def inverse_warp(img, H_inv, output_shape):
        return cv2.warpPerspective(img, H_inv, output_shape)

    def forward_warp_with_splatting(img, H, output_shape):
        h, w = img.shape[:2]
        new_h, new_w = output_shape
        warped = np.zeros((new_h, new_w, 3), dtype=np.float32)
        weight = np.zeros((new_h, new_w), dtype=np.float32)

        # Inverse for normalization
        for y in range(h):
            for x in range(w):
                pt = np.array([x, y, 1.0])
                warped_pt = H @ pt
                warped_pt /= warped_pt[2]  # Normalize

                u, v = warped_pt[:2]
                if 0 <= u < new_w - 1 and 0 <= v < new_h - 1:
                    u0, v0 = int(u), int(v)
                    du, dv = u - u0, v - v0

                    # Bilinear weights
                    w00 = (1 - du) * (1 - dv)
                    w01 = (1 - du) * dv
                    w10 = du * (1 - dv)
                    w11 = du * dv

                    pixel = img[y, x].astype(np.float32)

                    # Accumulate values and weights
                    warped[v0, u0] += pixel * w00
                    warped[v0+1, u0] += pixel * w01
                    warped[v0, u0+1] += pixel * w10
                    warped[v0+1, u0+1] += pixel * w11

                    weight[v0, u0] += w00
                    weight[v0+1, u0] += w01
                    weight[v0, u0+1] += w10
                    weight[v0+1, u0+1] += w11

        # Normalize by weight to get final pixel values
        nonzero = weight > 0
        warped[nonzero] /= weight[nonzero, None]

        return np.clip(warped, 0, 255).astype(np.uint8)


    # Input image and homography
    h, w = img1_rgb.shape[:2]

    # Warp corners to get new size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), h_matrix)
    [x_min, y_min] = np.min(warped_corners, axis=(0, 1))
    [x_max, y_max] = np.max(warped_corners, axis=(0, 1))

    # New output canvas size
    new_w = int(np.ceil(x_max - x_min))
    new_h = int(np.ceil(y_max - y_min))

    # Translate to ensure (0,0) origin
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    H_adj = T @ h_matrix
    H_inv = np.linalg.inv(H_adj)

    # Apply forward warping with splatting
    forward_result = forward_warp_with_splatting(img1_rgb, H_adj, (new_h, new_w))

    # Now inverse warp that result
    warped_image = inverse_warp(forward_result, H_inv, (w, h))

    def stitching (img2, warped_image1):
        # Create a Stitcher object
        stitcher = cv2.Stitcher_create()

        # Stitch the images
        (status, stitched) = stitcher.stitch([img2, warped_image1])

        # Check if stitching succeeded
        if status == cv2.Stitcher_OK:
            print("Stitching successful.")
        else:
            print("Stitching failed with status code:", status)
            
        return stitched
            
    stitched_image = stitching(img2_rgb, warped_image)
    return stitched_image
    
image1 = cv2.imread('img1.png')
stitched_image = cv2.imread('img1.png')
for i in range (2,3):
    img = cv2.imread(f'img{i}.png')
    if (img.shape):
        print("done!!")
    stitched_image = photosphere(img, stitched_image)
    
cv2.imshow("Stitched Image", stitched_image)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()