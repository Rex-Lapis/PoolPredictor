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
shape = None

setting = 0

# TODO: Try analyzing ratio of h & w of table


class Table:
    def __init__(self, setting=0):
        self.edges = {}
        self.frame = None
        self.playbounds = None
        self.bumperbounds = None
        self.tablebounds = None
        self.linelist = None
        self.setting = setting
        self.find_table_lines()
        self.set_boundaries()
        self.drawlines(color=(0, 200, 255))

        cv.imwrite('./debug_images/9_final_lines.png', self.frame)

    def find_table_lines(self):
        global shape
        if cap.isOpened():
            ret, frame = cap.read()
            self.frame = frame
        count = 0
        n_frames = 10
        self.linelist = []

        shape = frame.shape

        height, width = frame.shape[0], frame.shape[1]

        while count <= n_frames:
            ret, frame = cap.read()

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if self.setting == 0:
                canny = auto_canny(gray, 0.8, 0.5)
            elif self.setting == 1:
                # Manual Canny version
                thresh_ratio = 3
                low = 35
                canny = cv.Canny(frame, low, low * thresh_ratio)

            cv.imwrite('./debug_images/1_canny_check.png', canny)
            minlinelength = 1000000
            maxlinegap = 10
            lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 300, minlinelength, maxlinegap)  # thresh was 300
            if lines is not None:
                for i in range(len(lines)):
                    self.linelist.append(lines[i][0])
            count += 1

        cannywlines = self.drawlines(canny)
        cv.imwrite('./debug_images/2_canny_check_lines.png', cannywlines)

        print(len(self.linelist), 'lines found')
        poplist = []

        for i in range(len(self.linelist)):
            line = self.linelist[i]
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            if not (x2 - x1) == 0:  # makeing sure we don't get a divide by zero error
                slope = (y2 - y1) / (x2 - x1)  # rise over run
                # print(slope)
            if slope != 0:
                poplist.append(i)
            else:
                pass
        print(len(poplist), 'lines removed')

        clean1 = self.drawlines(self.frame.copy(), color=(0, 0, 255))
        cv.imwrite('./debug_images/3_1st_pass_clean.png', clean1)

        lencount = 0
        for i in range(len(poplist)):
            ind = poplist[i] - lencount
            lencount += 1
            del self.linelist[ind]

        horizontal, vertical = self.group_lines_by_direction(minlinelen=30)

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

        horiz = self.drawlines(frame.copy(), linelist=horizontal, color=(0, 0, 255))
        cv.imwrite('./debug_images/4_horizontal_split.png', horiz)
        vert = self.drawlines(frame.copy(), linelist=vertical, color=(0, 0, 255))
        cv.imwrite('./debug_images/4_vertical_split.png', vert)

        # copy2 = self.drawlines(frame.copy(), (0, 0, 255))
        # cv.imwrite('./debug_images/horizontal_vertical_sort.png', copy2)

        toplines = self.group_lines(toplines)
        bottomlines = self.group_lines(bottomlines)
        leftlines = self.group_lines(leftlines)
        rightlines = self.group_lines(rightlines)

        top = self.drawlines(frame.copy(), linelist=toplines, color=(0, 0, 255))
        cv.imwrite('./debug_images/5_toplines.png', top)
        bottom = self.drawlines(frame.copy(), linelist=bottomlines, color=(0, 0, 255))
        cv.imwrite('./debug_images/5_bottomlines.png', bottom)
        left = self.drawlines(frame.copy(), linelist=leftlines, color=(0, 0, 255))
        cv.imwrite('./debug_images/5_leftlines.png', left)
        right = self.drawlines(frame.copy(), linelist=rightlines, color=(0, 0, 255))
        cv.imwrite('./debug_images/5_rightlines.png', right)

        toplines = self.find_average_lines_from_groups(toplines, axis='y')
        bottomlines = self.find_average_lines_from_groups(bottomlines, axis='y')
        leftlines = self.find_average_lines_from_groups(leftlines, axis='x')
        rightlines = self.find_average_lines_from_groups(rightlines, axis='x')

        top = self.drawlines(frame.copy(), linelist=toplines, color=(0, 0, 255))
        cv.imwrite('./debug_images/6_toplines_avg.png', top)
        bottom = self.drawlines(frame.copy(), linelist=bottomlines, color=(0, 0, 255))
        cv.imwrite('./debug_images/6_bottomlines_avg.png', bottom)
        left = self.drawlines(frame.copy(), linelist=leftlines, color=(0, 0, 255))
        cv.imwrite('./debug_images/6_leftlines_avg.png', left)
        right = self.drawlines(frame.copy(), linelist=rightlines, color=(0, 0, 255))
        cv.imwrite('./debug_images/6_rightlines_avg.png', right)

        self.linelist = [toplines, rightlines, bottomlines, leftlines]
        self.linelist = {'top': toplines, 'bottom': bottomlines, 'left': leftlines, 'right': rightlines}

        copy2 = self.drawlines(frame.copy(), color=(0, 0, 255))
        cv.imwrite('./debug_images/7_grouped.png', copy2)

        print('Table Found')
        return frame

    def drawlines(self, frame=None, linelist=None, color=(0, 0, 255), thickness=2):
        if frame is None:
            frame = self.frame
        if linelist is None:
            linelist = self.linelist
        if len(frame.shape) == 2:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if isinstance(linelist, dict):
            for group in linelist.values():
                for line in group:
                    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                    pt1, pt2 = (x1, y1), (x2, y2)
                    cv.line(frame, pt1, pt2, color, thickness)
        elif isinstance(linelist, (list, np.ndarray)):
            linelist = list(linelist)
            for linelst in linelist:
                linelst = list(linelist)
                if isinstance(linelst[0], (int, float, np.int32)):
                    x1, y1, x2, y2 = linelst[0], linelst[1], linelst[2], linelst[3]
                    pt1, pt2 = (x1, y1), (x2, y2)
                    cv.line(frame, pt1, pt2, color, thickness)
                else:
                    for line in linelst:
                        if isinstance(line[0], (int, float, np.int32)):
                            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                            pt1, pt2 = (x1, y1), (x2, y2)
                            cv.line(frame, pt1, pt2, color, thickness)
                            print('line drawn')
            # self.linelist = np.asarray(self.linelist)
            # print(self.linelist.shape)
            # for group in self.linelist:
            #     print(group.shape)
                # for line in group:
                #     x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                #     pt1, pt2 = (x1, y1), (x2, y2)
                #     cv.line(frame, pt1, pt2, color, 2)\
        return frame

    def set_boundaries(self):
        lines = self.linelist
        lines = {key: np.asarray(value) for (key, value) in lines.items()}
        top, bottom = lines['top'], lines['bottom']
        left, right = lines['left'], lines['right']

        # x, y = 0, 1
        #
        # table_t = top[top.argmin(axis=0)[y]]
        # table_b = bottom[bottom.argmax(axis=0)[y]]
        # table_l = left[left.argmin(axis=0)[x]]
        # table_r = right[right.argmax(axis=0)[x]]
        # table_lines = [table_t, table_b, table_l, table_r]

        # playfield_t = top[top.argmax(axis=0)[y]]
        # playfield_b = bottom[bottom.argmin(axis=0)[y]]
        # playfield_l = left[left.argmax(axis=0)[x]]
        # playfield_r = right[right.argmin(axis=0)[x]]
        # playfield_lines = [playfield_t, playfield_b, playfield_l, playfield_r]

        bumper_lines = []
        # max, mid, low
        playfield_t, pockets_t, table_t = self.max_mid_min(top, axis='y')
        table_b, pockets_b, playfield_b = self.max_mid_min(bottom, axis='y')
        table_r, pockets_r, playfield_r = self.max_mid_min(right, axis='x')
        playfield_l, pockets_l, table_l = self.max_mid_min(left, axis='x')

        table_lines = [table_t, table_b, table_l, table_r]
        playfield_lines = [playfield_t, playfield_b, playfield_l, playfield_r]
        pocket_lines = [pockets_t, pockets_b, pockets_l, pockets_r]

        # for line in playfield_lines:
        #     x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        #     cv.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
        #
        # for line in table_lines:
        #     x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        #     cv.line(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        #
        # for line in pocket_lines:
        #     x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        #     cv.line(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        playfield = self.drawlines(self.frame.copy(), playfield_lines, (0, 0, 255), 5)
        cv.imwrite('./debug_images/8_border_playfield.png', playfield)

        table_ = self.drawlines(self.frame.copy(), table_lines, (255, 0, 0), 5)
        cv.imwrite('./debug_images/8_border_table.png', table_)

        pockets = self.drawlines(self.frame.copy(), pocket_lines, (0, 255, 0), 5)
        cv.imwrite('./debug_images/8_border_pockets.png', pockets)

    def max_mid_min(self, group, axis='x'):
        if axis == 'x':
            xory = 0
        elif axis == 'y':
            xory = 1
        # print(len(group))
        group2 = list(group)
        maxind = group.argmax(axis=0)[xory]
        maximum = group2[maxind]
        group2.pop(maxind)
        group2 = np.asarray(group2)
        minind = group2.argmin(axis=0)[xory]
        minimum = group2[minind]
        group2 = list(group2)
        group2.pop(minind)
        midimum = group2[0]
        # print(maximum, midimum, minimum)

        return maximum, midimum, minimum

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

    def find_average_lines_from_groups(self, groups, axis='x'):
        output = []
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

    def group_lines_by_direction(self, minlinelen=0):
        vertical = []
        horizontal = []
        for i in range(len(self.linelist)):
            line = self.linelist[i]
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            pt1, pt2 = (x1, y1), (x2, y2)
            # cv.line(frame, pt1, pt2, (255, 0, 0), 2)
            if x1 == x2 and abs(y1 - y2) > minlinelen:
                vertical.append(self.linelist[i])
            elif y1 == y2 and abs(x1 - x2) > minlinelen:
                horizontal.append(self.linelist[i])
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

#
# class Pocket:
#     def __init__(self):
#         pass
#
#
# class Ball:
#     def __init__(self):
#         pass
#
#
# class Stick:
#     def __init__(self):
#         pass


def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


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
        table = Table(setting=setting)
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
