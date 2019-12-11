import cv2 as cv
import numpy as np
import imutils
import cython
import time

cv2 = cv

cap = cv.VideoCapture('2019_PoolChamp_Clip7.mp4')
table_color_bgr = (60, 105, 0)
table_color_thresh = 30
ratio = (2, 1)
shape = None

# TODO: Try analyzing ratio of h & w of table


class Table:
    def __init__(self):
        self.edges = {}
        self.frame = None
        self.playbounds = None
        self.bumperbounds = None
        self.tablebounds = None
        self.linelist = None
        self.find_table_lines()
        self.set_boundaries()
        self.drawlines(color=(0, 200, 255))

        cv.imwrite('./debug_images/lineimg1.png', self.frame)

    def find_table_lines(self):
        global shape
        if cap.isOpened():
            ret, frame = cap.read()
            self.frame = frame
        count = 0
        n_frames = 10
        linelist = []

        shape = frame.shape

        height, width = frame.shape[0], frame.shape[1]

        while count <= n_frames:
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
            if not (x2 - x1) == 0:  # makeing sure we don't get a divide by zero error
                slope = (y2 - y1) / (x2 - x1)  # rise over run
            if slope != 0:
                poplist.append(i)
            else:
                pass

        lencount = 0
        for i in range(len(poplist)):
            ind = poplist[i] - lencount
            lencount += 1
            del linelist[ind]

        horizontal, vertical = self.group_lines_by_direction(linelist, minlinelen=30)

        toplines, bottomlines = [], []
        leftlines, rightlines = [], []

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

        toplines = self.group_lines(toplines)
        bottomlines = self.group_lines(bottomlines)
        leftlines = self.group_lines(leftlines)
        rightlines = self.group_lines(rightlines)

        toplines = self.find_average_lines_from_groups(toplines)
        bottomlines = self.find_average_lines_from_groups(bottomlines)
        leftlines = self.find_average_lines_from_groups(leftlines)
        rightlines = self.find_average_lines_from_groups(rightlines)

        self.linelist = [toplines, rightlines, bottomlines, leftlines]
        self.linelist = {'top': toplines, 'bottom': bottomlines, 'left': leftlines, 'right': rightlines}

        print('Table Found')
        return frame

    def drawlines(self, frame=None, color=(0, 0, 255)):
        if frame is None:
            frame = self.frame
        for group in self.linelist.values():
            for line in group:
                x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                pt1, pt2 = (x1, y1), (x2, y2)
                cv.line(frame, pt1, pt2, color, 2)

    def set_boundaries(self):
        lines = self.linelist
        lines = {key: np.asarray(value) for (key, value) in lines.items()}
        top, bottom = lines['top'], lines['bottom']
        left, right = lines['left'], lines['right']
        x, y = 0, 1
        bumper_lines =[]

        table_t = top[top.argmin(axis=0)[y]]
        table_b = bottom[bottom.argmax(axis=0)[y]]
        table_l = left[left.argmin(axis=0)[x]]
        table_r = right[right.argmax(axis=0)[x]]

        table_lines = [table_t, table_b, table_l, table_r]

        bumper_t = top[top.argmax(axis=0)[y]]
        bumper_b = bottom[bottom.argmin(axis=0)[y]]
        bumper_l = left[left.argmax(axis=0)[x]]
        bumper_r = right[right.argmin(axis=0)[x]]
        bumber_edges = [bumper_t, bumper_b, bumper_l, bumper_r]
        print(top)
        print(bottom)

        print('t_bump', bumper_t)
        print('b_bump', bumper_b)
        print('l_bump', bumper_l)
        print('r_bump', bumper_r)

        for line in bumber_edges:
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            cv.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

        for line in table_lines:
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            cv.line(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 5)


    def find_contours_from_lines(self, linelist):
        print('\n')
        axis = self.check_axis(linelist[0])
        print('len:', len(linelist))
        print('axis:', axis)
        outlist = []
        for group in linelist:
            xy_list = []
            for line in group:
                xy_list.append([line[0], line[1]])
                xy_list.append([line[2], line[3]])
                # print(line)
            if axis == 'y':
                # Sort by x
                outlist.append(np.sort(xy_list, 0))
            elif axis == 'x':
                outlist.append(np.sort(xy_list, 1))
            print(xy_list)
        return np.asarray(outlist)

    def find_average_lines_from_groups(self, groups):
        output = []
        axis = self.check_axis(groups[0])
        # group = np.asarray(groups)
        for group in groups:
            group = np.asarray(group)
            if axis == 'x':
                avgx = int(np.mean(group[:, 0]))
                newline = [avgx, 0, avgx, shape[0]]
            elif axis == 'y':
                avgy = int(np.mean(group[:, 1]))
                newline = [0, avgy, shape[1], avgy]
            output.append(newline)
        return output

    def group_lines_by_direction(self, linelist, minlinelen=0):
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

    def group_lines(self, linelist, thresh=20):
        groups = []
        axis = self.check_axis(linelist)
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

    def check_axis(self, linelist):
        linelist = np.asarray(linelist)
        xvariance = np.std(linelist[:, 0])
        yvariance = np.std(linelist[:, 1])
        if xvariance < yvariance:
            axis = 'x'
        else:
            axis = 'y'
        return axis

    def find_balls(self, frame):
        copy = frame.copy()
        gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        min_dist = frame.shape[0] / 64
        max_radius = round(frame.shape[0] / 22.5)  # was / 22.5

        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, min_dist, param1=53, param2=30, minRadius=5,
                                  maxRadius=max_radius)
        radiuslist = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y = circle[0], circle[1]
                radius = circle[2]
                radiuslist.append(radius)
                cv.circle(frame, (x, y), radius, (0, 255, 0), 2)
                cv.circle(frame, (x, y), 2, (0, 0, 255), 3)
        circle_img = frame
        # print('circle radius max:', str(max(radiuslist)), 'min:', str(min(radiuslist)))
        return circle_img


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


def check_contour_shape(contour):
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
    max_radius = round(img.shape[0]/22.5)  # was / 22.5

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, min_dist, param1=53, param2=30, minRadius=5, maxRadius=max_radius)
    radiuslist = []
    if circles is not None:
        print(len(circles), 'circles')
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y = circle[0], circle[1]
            print(x, y)
            radius = circle[2]
            radiuslist.append(radius)
            cv.circle(img, (x, y), radius, (0, 255, 0), 2)
            cv.circle(img, (x, y), 2, (0, 0, 255), 3)
    circle_img = img
    # print('circle radius max:', str(max(radiuslist)), 'min:', str(min(radiuslist)))
    return circle_img


# def set_table_boundaries():
#     global shape
#     if cap.isOpened():
#         ret, frame = cap.read()
#     count = 0
#     n_frames = 10
#     linelist = []
#
#     shape = frame.shape
#
#     height, width = frame.shape[0], frame.shape[1]
#
#     while count <= n_frames:
#         ret, frame = cap.read()
#         # Manual Canny version in comment below
#         # thresh_ratio = 3
#         # low = 35
#         # canny = cv.Canny(frame, low, low * thresh_ratio)  #
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         canny = auto_canny(gray, 0.8, 0.5)
#         cv.imwrite('./debug_images/outline_check.png', canny)
#
#         minlinelength = 1000000
#         maxlinegap = 10
#         lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 300, minlinelength, maxlinegap)  # thresh was 300
#         if lines is not None:
#             for i in range(len(lines)):
#                 linelist.append(lines[i][0])
#         count += 1
#
#     print(len(linelist), 'lines found')
#     poplist = []
#
#     for i in range(len(linelist)):
#         line = linelist[i]
#         x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
#         if not (x2 - x1) == 0:                    # makeing sure we don't get a divide by zero error
#             slope = (y2 - y1) / (x2 - x1)         # rise over run
#         if slope != 0:
#             poplist.append(i)
#         else:
#             pass
#
#     lencount = 0
#     for i in range(len(poplist)):
#         ind = poplist[i] - lencount
#         lencount += 1
#         del linelist[ind]
#
#     horizontal, vertical = group_lines_by_direction(linelist, minlinelen=30)
#
#     toplines, bottomlines = [], []
#     leftlines, rightlines = [], []
#
#     # Split horizontal into top and bottom
#     for i in range(len(horizontal)):
#         line = horizontal[i]
#         x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
#         ydim = frame.shape[0]
#         if 5 < y1 < ydim / 4:
#             toplines.append(line)
#         elif (height - 5) > y1 > (ydim - (ydim / 4)):
#             bottomlines.append(line)
#     # Split vertical into right and left
#     for i in range(len(vertical)):
#         line = vertical[i]
#         x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
#         xdim = frame.shape[1]
#         if 5 < x1 < xdim / 4:
#             leftlines.append(line)
#         elif (width - 5) > x1 > (xdim - (xdim / 4)):
#             rightlines.append(line)
#
#     toplines = group_lines(toplines)
#     bottomlines = group_lines(bottomlines)
#     leftlines = group_lines(leftlines)
#     rightlines = group_lines(rightlines)
#
#     toplines = find_average_lines_from_groups(toplines)
#     bottomlines = find_average_lines_from_groups(bottomlines)
#     leftlines = find_average_lines_from_groups(leftlines)
#     rightlines = find_average_lines_from_groups(rightlines)
#
#     horizontal = [toplines, bottomlines]
#     vertical = [leftlines, rightlines]
#
#     linelist = [horizontal, vertical]
#     linelist = [toplines, rightlines, bottomlines, leftlines]
#
#     # for group in topcontours:
#     #     # for contour in group:
#     #     group = np.asarray(group)
#     #     cv.drawContours(frame, [group], -1, (255, 0, 0), 2)
#
#
#     for group in linelist:
#         for line in group:
#             x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
#             pt1, pt2 = (x1, y1), (x2, y2)
#             # cv.line(frame, pt1, pt2, (255, 0, 0), 2)
#             cv.line(frame, pt1, pt2, (0, 255, 0), 2)
#
#
#
#     # Draw out all lines for test image
#     # for orientation in linelist:
#     #     for group in orientation:
#     #         for lines in group:
#     #             for i in lines:
#     #                 line = i
#     #                 x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
#     #                 pt1, pt2 = (x1, y1), (x2, y2)
#     #                 # cv.line(frame, pt1, pt2, (255, 0, 0), 2)
#     #                 cv.line(frame, pt1, pt2, (0, 255, 0), 2)
#     print('Table Found')
#     return frame


def play_frame():
    ret, frame = cap.read()
    if ret:
        table.drawlines(frame)
        table.find_balls(frame)
        # frame = draw_circles(frame)
        cv.imshow('frame', frame)
        # frame = crop(frame)
#         find_table(frame)
#         # circles = draw_circles(frame)
#         cv.imshow('frame', frame)
#         # cv.imshow('circles', circles)
#         # cv.moveWindow('circles', 20, 20)
        if cv.waitKey(25) & 0xFF == 27:  # 27 is the esc key's number
            return False
        else:
            return True
    else:
        return False


def main():
    global table
    if cap.isOpened():
        table = Table()
    else:
        print("error opening video")
    while cap.isOpened():
        playing = play_frame()
        if not playing:
            break

    cap.release()
    cv.destroyAllWindows()


main()

# shape is h x w x chan
