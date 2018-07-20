import cv2
import numpy as np

def convert_to_normalized_bw(hsv, mask):
    object_pixels = hsv[mask == 255][:,2]
    min, max = np.min(object_pixels), np.max(object_pixels)

    hsv[mask == 255] = (hsv[mask == 255] - [0,0,min]) * [0,0,255/(max - min)]
    hsv[:,:,1] = 0
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

input = cv2.imread('red.png')
frame=cv2.GaussianBlur(input, (3, 3), 0)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

colorMin = (0, 50, 0)
colorMax = (360, 255, 255)
mask = cv2.inRange(hsv, colorMin, colorMax)

# mask = cv2.dilate(mask, None, iterations=10)
frame[mask!=255] = 0
normalized_bw = convert_to_normalized_bw(frame, mask)

_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
border = {'x': 2, 'y': 2}
x,y,w,h = cv2.boundingRect(contours[0])
# cv2.imwrite('out.png', cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2))

side_length = max(h,w)
cv2.imwrite('out.png', normalized_bw[y-border['x']:y+side_length+border['x'], x-border['x']:x+side_length+border['x']])

# cv2.imwrite('out.png', cv2.drawContours(res, contours, 0, (255,0,0), 3))
