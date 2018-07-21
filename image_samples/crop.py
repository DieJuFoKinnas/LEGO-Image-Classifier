import cv2
import numpy as np


def gen_mask(hsv, threashold):
    colorMin = (0, threashold, 0)
    colorMax = (360, 255, 255)
    cv2.inRange(hsv, colorMin, colorMax)

    return cv2.inRange(hsv, colorMin, colorMax)


def convert_to_normalized_bw(img, threashold):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = gen_mask(hsv, threashold)

    object_pixels = hsv[mask == 255][:,2]
    min, max = np.min(object_pixels), np.max(object_pixels)
    hsv[mask == 255] = (hsv[mask == 255] - [0,0,min]) * [0,0,255/(max - min)]

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray[mask!=255] = 255

    return gray


def get_bounding_box(img, threashold, border_x=2, border_y=2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = gen_mask(hsv, threashold)
    #cv2.imwrite('mask.png', mask)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imwrite('contours.png', cv2.drawContours(hsv, contours, 0, (255,0,0), 3))

    x,y,w,h = cv2.boundingRect(contours[0])
    side_length = max(w,h)

    return x - border_x, x + side_length + border_x, y - border_y, y + side_length + border_y


def crop(img, threashold, blurr_size=31, border_x=2, border_y=2):
    blurred = cv2.GaussianBlur(img, (blurr_size, blurr_size), 0)
    x_min, x_max, y_min, y_max = get_bounding_box(blurred, border_x=border_x, border_y=border_y, threashold=threashold)

    return img[y_min:y_max, x_min:x_max]


def process_img(img_name, threashold=100):
    img = cv2.imread(img_name)
    # downsample so the too high res shadows don't cause problems for the bounding box
    img = cv2.resize(img, (1000, 750), interpolation=cv2.INTER_CUBIC)
    cropped = crop(img, threashold)

    # sample to fixed image size
    resized = cv2.resize(cropped, (100, 100), interpolation=cv2.INTER_CUBIC)
    normalized = convert_to_normalized_bw(resized, threashold)

    cv2.imwrite('out.jpg', normalized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


process_img('foto3.jpg', threashold=100)
