import cv2
import numpy as np
import matplotlib.pyplot as plt

# Simple manual white balance (gray world assumption)
def simple_white_balance(img):
    result = img.copy()
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_gray / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_gray / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_gray / avg_r), 0, 255)

    return result.astype(np.uint8)


img = cv2.imread("RTSPCamera6_frame_066.jpg")
enhanced_img = simple_white_balance(img)

cv2.imwrite("RTSPCamera6_frame_066.jpg", enhanced_img)
cv2.imshow("enhanced_iamge", enhanced_img)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()