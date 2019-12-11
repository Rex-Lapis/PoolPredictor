import cv2 as cv
import numpy as np
import imutils
import cython
import time

cv2 = cv

cap = cv.VideoCapture('2019_PoolChamp_Clip4.mp4')
table_color_bgr = (60, 105, 0)
table_color_thresh = 30
ratio = (2, 1)
# %%cython -a

# TODO: Try analyzing ratio of h & w of table


class Table:
    def __init__(self):
        pass


class Pocket:
    def __init__(self):
        pass


class Ball:
    def __init__(self):
        pass


class Stick:
    def __init__(self):
        pass


def crop(frame, xcrop=0.05, ycrop=0.05):
    shape = frame.shape
    xcrop = round(shape[0] * xcrop)
    ycrop = round(shape[1] * ycrop)
    frame = frame[xcrop: -xcrop, ycrop: -ycrop]
    return frame


def color_difference(col1, col2):
    diff = 0
    for i in range(3):
        diff += abs(col1[i] - col2[i])
    return diff


def auto_canny(image, sigma=0.33, uppermod=1, lowermod=1):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v) * lowermod)
    upper = int(min(255, (1.0 + sigma) * v) * uppermod)
    # print(lower, upper)
    edged = cv.Canny(image, lower, upper)
    return edged


def find_the_green(img, thresh=50):
    x1, y1, x2, y2 = None, None, None, None
    brk = False
    for row in range(len(img)):
        for pixel in range(len(img[row])):
            diff = color_difference(img[row][pixel], table_color_bgr)
            diff2 = color_difference(img[-row][-pixel], table_color_bgr)
            # print('diff1: ', diff, 'diff2: ', diff2)
            if diff < thresh:
                x1 = pixel
                y1 = row
            if diff2 < thresh:
                x2 = img.shape[1] - pixel
                y2 = img.shape[0] - row
            if x1 and y1 and x2 and y2:
                print('done with loop', row)
                brk = True
                break
        if brk:
            break
    return (x1, y1), (x2, y2)


def find_table(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # pt1, pt2 = find_the_green(img, thresh=30)
    # print(pt1, pt2)
    # cv.rectangle(img, pt1, pt2, (255, 0, 0), 2)
    # blurred = cv.bilateralFilter(gray, 9, 75, 75)
    # blurred = gray

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    # wide = cv2.Canny(blurred, 49, 98)
    # tight = cv2.Canny(blurred, 225, 250)
    # ret, thresh = cv.threshold(img, 98, 255, cv2.THRESH_BINARY)
    # auto = auto_canny(gray)
    ratio = 3
    low = 35
    canny = cv.Canny(img, low, low*ratio)

    minLineLength = 1000000
    maxLineGap = 10
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 250, minLineLength, maxLineGap)
    if lines is not None:
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                pass
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # show the images
    cv2.imshow("Edges", canny)
    cv2.imshow("Original", img)


    # cv2.imshow('Thresh', thresh)

    # cv2.waitKey(0)

    # blank = np.zeros(img.shape)

    # img[:,:,2] = 0
    # blur = cv.GaussianBlur(img, (3, 3), 0)
    # gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    # lower, upper = calculate_canny_thresholds(img, sigma=0.33)
    # edged2 = cv.Canny(img, lower, upper)
    # cv.imshow('img', img)
    # cv.imshow('Edges_gb', edged2)
    # cv.moveWindow('Edges_gb', 200, 200)
    # edges = cv.Canny(img, 100, 200)
    # contours, hierarchy = cv.findContours(blurred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    # # cv.imshow('edges', edges)
    #
    # if len(contours) > 0:
    #     for contour in range(len(contours)):
    #         cv.drawContours(blank, contours, contour, (255, 0, 255), 1)
    #         # cv.imshow('edges', edges)
    #     cv.imshow('contours', blank)


def fast(image):
    img = image.copy()
    height, width, depth = img.shape
    img[0:height, 0:width // 4, 0:depth] = 0  # DO THIS INSTEAD
    return img


def detect_table(contour):
    shape = None
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.04 * peri, True)

    if len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"


def draw_circles(img):
    img = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    min_dist = img.shape[0]/64
    max_radius = round(img.shape[0]/22.5)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, min_dist, param1=53, param2=30, minRadius=5, maxRadius=max_radius)
    radiuslist = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y = circle[0], circle[1]
            radius = circle[2]
            radiuslist.append(radius)
            cv.circle(img, (x, y), radius, (0, 255, 0), 2)
            cv.circle(img, (x, y), 2, (0, 0, 255), 3)
    circle_img = img
    # print('circle radius max:', str(max(radiuslist)), 'min:', str(min(radiuslist)))
    return circle_img


def set_table_boundaries():

    if cap.isOpened():
        ret, frame = cap.read()
    count = 0
    linelist = []

    height, width = frame.shape[0], frame.shape[1]

    while cap.isOpened():
        if count > 10:
            break
        ret, frame = cap.read()
        # Manual Canny version in comment below
        # thresh_ratio = 3
        # low = 35
        # canny = cv.Canny(frame, low, low * thresh_ratio)  #
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        canny = auto_canny(gray, 0.8, 0.5)
        cv.imwrite('./debug_images/outline_check.png', canny)

        minlinelength = 1000000
        maxlinegap = 10
        lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 300, minlinelength, maxlinegap)  # thresh was 300
        if lines is not None:
            for i in range(len(lines)):
                linelist.append(lines[i][0])
        count += 1

    print(len(linelist), 'lines found')
    poplist = []

    for i in range(len(linelist)):
        line = linelist[i]
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        if not (x2 - x1) == 0:                    # makeing sure we don't get a divide by zero error
            slope = (y2 - y1) / (x2 - x1)         # rise over run
        if slope != 0:
            poplist.append(i)
        else:
            pass

    lencount = 0
    for i in range(len(poplist)):
        ind = poplist[i] - lencount
        lencount += 1
        del linelist[ind]

    toplines, bottomlines = [], []
    leftlines, rightlines = [], []

    horizontal, vertical = group_lines_by_direction(linelist, minlinelen=30)

    # Categorize into vertical and horizontal lines
    # for i in range(len(linelist)):
    #     line = linelist[i]
    #     x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    #     pt1, pt2 = (x1, y1), (x2, y2)
    #     # cv.line(frame, pt1, pt2, (255, 0, 0), 2)
    #     if x1 == x2 and abs(y1 - y2) > minlinelen:
    #         vertical.append(linelist[i])
    #     elif y1 == y2 and abs(x1 - x2) > minlinelen:
    #         horizontal.append(linelist[i])

    # Split horizontal into top and bottom
    for i in range(len(horizontal)):
        line = horizontal[i]
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        ydim = frame.shape[0]
        if 5 < y1 < ydim / 4:
            toplines.append(line)
        elif (height - 5) > y1 > (ydim - (ydim / 4)):
            bottomlines.append(line)
    # Split vertical into right and left
    for i in range(len(vertical)):
        line = vertical[i]
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        xdim = frame.shape[1]
        if 5 < x1 < xdim / 4:
            leftlines.append(line)
        elif (width - 5) > x1 > (xdim - (xdim / 4)):
            rightlines.append(line)

    # Test print a group of lines
    # for i in bottomlines:
    #     line = i
    #     x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    #     pt1, pt2 = (x1, y1), (x2, y2)
    #     # cv.line(frame, pt1, pt2, (255, 0, 0), 2)
    #     cv.line(frame, pt1, pt2, (0, 255, 0), 2)

    toplines = group_lines(toplines, axis='y')
    bottomlines = group_lines(bottomlines, axis='y')
    leftlines = group_lines(leftlines, axis='x')
    rightlines = group_lines(rightlines, axis='x')

    horizontal = [toplines, bottomlines]
    vertical = [leftlines, rightlines]

    # Sort toplines and bottomlines into height groups
    # for j in range(len(horizontal)):
    #     line_region = horizontal[j]
    #     groups = []
    #     for line in line_region:
    #         y1 = line[1]
    #         grouped = False
    #         for i in range(len(groups)):
    #             sample_x = groups[i][0][1]
    #             # print(abs(y1 - sample_x))
    #             if abs(y1 - sample_x) < thresh:
    #                 groups[i].append(line)
    #                 grouped = True
    #         if grouped is False:
    #             groups.append([line])
    #     if j == 0:
    #         toplines = groups
    #     elif j == 1:
    #         bottomlines = groups

    # Sort leftlines and rightlines into height groups
    # for j in range(len(vertical)):
    #     line_region = vertical[j]
    #     groups = []
    #     for line in line_region:
    #         x1 = line[0]
    #         grouped = False
    #         for i in range(len(groups)):
    #             sample_x = groups[i][0][0]
    #             # print(abs(x1 - sample_x))
    #             if abs(x1 - sample_x) < thresh:
    #                 groups[i].append(line)
    #                 grouped = True
    #         if grouped is False:
    #             groups.append([line])
    #     if j == 0:
    #         leftlines = groups
    #     elif j == 1:
    #         rightlines = groups

    print('\n')
    print(len(toplines))
    print(len(bottomlines))
    print(len(leftlines))
    print(len(rightlines))

    return frame


def group_lines_by_direction(linelist, minlinelen=0):
    vertical = []
    horizontal = []
    for i in range(len(linelist)):
        line = linelist[i]
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        pt1, pt2 = (x1, y1), (x2, y2)
        # cv.line(frame, pt1, pt2, (255, 0, 0), 2)
        if x1 == x2 and abs(y1 - y2) > minlinelen:
            vertical.append(linelist[i])
        elif y1 == y2 and abs(x1 - x2) > minlinelen:
            horizontal.append(linelist[i])
    return horizontal, vertical


def group_lines(linelist, thresh=20, axis='y'):
    groups = []
    for line in linelist:
        x1, y1 = line[0], line[1]
        grouped = False
        for i in range(len(groups)):
            if axis == 'y':
                sample_y = groups[i][0][1]
                # print(abs(y1 - sample_x))
                if abs(y1 - sample_y) < thresh:
                    groups[i].append(line)
                    grouped = True
            if axis == 'x':
                sample_x = groups[i][0][0]
                # print(abs(y1 - sample_x))
                if abs(x1 - sample_x) < thresh:
                    groups[i].append(line)
                    grouped = True
        if grouped is False:
            groups.append([line])

    return groups


def main():
    if cap.isOpened():
        lineimg = set_table_boundaries()
        cv.imwrite('./debug_images/lineimg60.2.png', lineimg)
    else:
        print("error opening video")
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         frame = crop(frame)
    #         find_table(frame)
    #         # circles = draw_circles(frame)
    #         # cv.imshow('frame', frame)
    #         # cv.imshow('circles', circles)
    #         # cv.moveWindow('circles', 20, 20)
    #         if cv.waitKey(25) & 0xFF == 27:  # 27 is the esc key's number
    #             break
    #     else:
    #         break
    cap.release()
    cv.destroyAllWindows()


main()

# shape is h x w x chan
