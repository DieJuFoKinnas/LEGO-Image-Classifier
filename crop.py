import cv2
import numpy as np

input = cv2.imread('renders/556-0.png')
# input = cv2.imread('renders/3195-0.png')

cv2.imwrite('out.png', input)
frame=cv2.GaussianBlur(input, (3, 3), 0)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

colorMin = (0, 50, 0)
colorMax = (360, 255, 255)

mask = cv2.inRange(hsv, colorMin, colorMax)


# WIP
object_pixels = input[mask == 255][:,2]
print(np.mean(object_pixels), np.std(object_pixels))

res = cv2.bitwise_and(frame, frame, mask=mask)

# mask = cv2.dilate(mask, None, iterations=10)

image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (255,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

border = {'x': 2, 'y': 2}
x,y,w,h = cv2.boundingRect(contours[0])
# cv2.imwrite('out.png', cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2))

side_length = max(h+border['x'], w+border['y'])
cv2.imwrite('out.png', input[y-border['y']: y+side_length, x-border['x']: x+side_length])

# cv2.imwrite('out.png', cv2.drawContours(res, contours, 0, (255,0,0), 3))
# cv2.imwrite('out.png', image)
