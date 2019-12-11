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

setting = 2

# TODO: Try analyzing ratio of h & w of table


class Table:
    def __init__(self, setting_num=0):
        self.edges = {}
        self.frame = None
        self.linelist = None
        self.playfieldlines = None
        self.pocketlines = None
        self.tablelines = None
        self.playbox = None
        self.pocketbox = None
        self.tablebox = None
        self.setting = setting_num
        self.find_table_lines()
        self.group_lines_by_category()
        self.set_boxes()
        self.drawlines(color=(0, 200, 255))
        cv.imwrite('./debug_images/10_final_lines.png', self.frame)

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

                minlinelength = 1000000
                maxlinegap = 10
                rho = 1
            elif self.setting == 1:
                # Manual Canny version
                thresh_ratio = 3
                low = 35
                canny = cv.Canny(frame, low, low * thresh_ratio)

                minlinelength = 1000000
                maxlinegap = 10
                rho = 1
            elif self.setting == 2:
                thresh_ratio = 3
                low = 30
                canny = cv.Canny(frame, low, low * thresh_ratio)
                # canny = auto_canny(gray, 0.8, 0.5)

                minlinelength = 5
                maxlinegap = 1
                rho = 1.5
            else:
                print('setting value not yet set')
                raise ValueError

            cv.imwrite('./debug_images/1_canny_check.png', canny)

            lines = cv2.HoughLinesP(canny, rho, np.pi / 180, 300, minLineLength=minlinelength, maxLineGap=maxlinegap)  # thresh was 300
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
            if slope != 0:
                poplist.append(i)
        self.remove_list_of_indexes(self.linelist, poplist)
        print(len(poplist), 'lines removed')

        clean1 = self.drawlines(self.frame.copy(), color=(0, 0, 255))
        cv.imwrite('./debug_images/3_slope_filter.png', clean1)

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

        self.linelist = {'top': toplines, 'bottom': bottomlines, 'left': leftlines, 'right': rightlines}

        self.group_lines_by_proximity()
        # toplines = self.group_lines_by_proximity(toplines)
        # bottomlines = self.group_lines_by_proximity(bottomlines)
        # leftlines = self.group_lines_by_proximity(leftlines)
        # rightlines = self.group_lines_by_proximity(rightlines)
        #
        # top = self.drawlines(frame.copy(), linelist=toplines, color=(0, 0, 255))
        # cv.imwrite('./debug_images/5_toplines.png', top)
        # bottom = self.drawlines(frame.copy(), linelist=bottomlines, color=(0, 0, 255))
        # cv.imwrite('./debug_images/5_bottomlines.png', bottom)
        # left = self.drawlines(frame.copy(), linelist=leftlines, color=(0, 0, 255))
        # cv.imwrite('./debug_images/5_leftlines.png', left)
        # right = self.drawlines(frame.copy(), linelist=rightlines, color=(0, 0, 255))
        # cv.imwrite('./debug_images/5_rightlines.png', right)

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

        # self.linelist = [toplines, rightlines, bottomlines, leftlines]
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
                if isinstance(group[0], (int, float, np.int32, np.int64)):
                    x1, y1, x2, y2 = group[0], group[1], group[2], group[3]
                    pt1, pt2 = (x1, y1), (x2, y2)
                    cv.line(frame, pt1, pt2, color, thickness)
                else:
                    for line in group:
                        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                        pt1, pt2 = (x1, y1), (x2, y2)
                        cv.line(frame, pt1, pt2, color, thickness)
        elif isinstance(linelist, (list, np.ndarray)):
            for lines in linelist:
                if isinstance(lines[0], (int, float, np.int32, np.int64)):
                    x1, y1, x2, y2 = lines[0], lines[1], lines[2], lines[3]
                    pt1, pt2 = (x1, y1), (x2, y2)
                    cv.line(frame, pt1, pt2, color, thickness)
                else:
                    for line in lines:
                        if isinstance(line[0], (int, float, np.int32, np.int64)):
                            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                            pt1, pt2 = (x1, y1), (x2, y2)
                            cv.line(frame, pt1, pt2, color, thickness)
            # self.linelist = np.asarray(self.linelist)
            # print(self.linelist.shape)
            # for group in self.linelist:
            #     print(group.shape)
                # for line in group:
                #     x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                #     pt1, pt2 = (x1, y1), (x2, y2)
                #     cv.line(frame, pt1, pt2, color, 2)\
        return frame

    def group_lines_by_category(self):
        lines = self.linelist
        lines = {key: np.asarray(value) for (key, value) in lines.items()}

        self.check_line_ratios(lines)

        top, bottom = lines['top'], lines['bottom']
        left, right = lines['left'], lines['right']

        playfield_t, pockets_t, table_t = self.max_mid_min(top, axis='y')
        table_b, pockets_b, playfield_b = self.max_mid_min(bottom, axis='y')
        table_r, pockets_r, playfield_r = self.max_mid_min(right, axis='x')
        playfield_l, pockets_l, table_l = self.max_mid_min(left, axis='x')

        table_lines = {'top': table_t, 'bottom': table_b, 'left': table_l, 'right': table_r}
        playfield_lines = {'top': playfield_t, 'bottom': playfield_b, 'left': playfield_l, 'right': playfield_r}
        pocket_lines = {'top': pockets_t, 'bottom': pockets_b, 'left': pockets_l, 'right': pockets_r}
        self.tablelines = table_lines
        self.playfieldlines = playfield_lines
        self.pocketlines = pocket_lines

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

        playfield = self.drawlines(self.frame.copy(), playfield_lines, (0, 0, 255), 5)  # TODO THESE LINES AREN'T WORKING. START HERE @@@@@@@@@@@@
        cv.imwrite('./debug_images/8_border_playfield.png', playfield)

        table_ = self.drawlines(self.frame.copy(), table_lines, (255, 0, 0), 5)
        cv.imwrite('./debug_images/8_border_table.png', table_)

        pockets = self.drawlines(self.frame.copy(), pocket_lines, (0, 255, 0), 5)
        cv.imwrite('./debug_images/8_border_pockets.png', pockets)

    def check_line_ratios(self, linedict=None):
        if linedict is None:
            linedict = self.linelist
        good = []
        for key in linedict:
            group = linedict[key]
            if len(group) == 3:
                good.append(group)
        print(len(good))
        print(good)

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

    def group_lines_by_proximity(self, linelist=None, thresh=30, group_num_thresh=10):

        def group(line_):
            x1, y1 = line_[0], line_[1]
            grouped = False
            for i in range(len(groups)):
                if axis == 'y':
                    sample_y = groups[i][0][1]
                    if abs(y1 - sample_y) < thresh:
                        groups[i].append(line_)
                        grouped = True
                if axis == 'x':
                    sample_x = groups[i][0][0]
                    if abs(x1 - sample_x) < thresh:
                        groups[i].append(line_)
                        grouped = True
            if grouped is False:
                groups.append([line_])

        setlinelist = False

        if linelist is None:
            linelist = self.linelist
            setlinelist = True
            #
            # toplines = self.group_lines_by_proximity(toplines)
            # bottomlines = self.group_lines_by_proximity(bottomlines)
            # leftlines = self.group_lines_by_proximity(leftlines)
            # rightlines = self.group_lines_by_proximity(rightlines)
            #
            # top = self.drawlines(frame.copy(), linelist=toplines, color=(0, 0, 255))
            # cv.imwrite('./debug_images/5_toplines.png', top)
            # bottom = self.drawlines(frame.copy(), linelist=bottomlines, color=(0, 0, 255))
            # cv.imwrite('./debug_images/5_bottomlines.png', bottom)
            # left = self.drawlines(frame.copy(), linelist=leftlines, color=(0, 0, 255))
            # cv.imwrite('./debug_images/5_leftlines.png', left)
            # right = self.drawlines(frame.copy(), linelist=rightlines, color=(0, 0, 255))
            # cv.imwrite('./debug_images/5_rightlines.png', right)

        if isinstance(linelist, dict):
            for key in linelist:
                if key == 'top' or key == 'bottom':
                    axis = 'y'
                else:
                    axis = 'x'
                groups = []
                list_ = linelist[key]
                for line in list_:
                    group(line)
                if setlinelist:
                    print(len(groups))
                    linelist[key] = groups

        elif isinstance(linelist, (list, np.ndarray)):
            axis = self.check_axis(linelist)
            groups = []
            for line in linelist:
                group(line)
                # x1, y1 = line[0], line[1]
                # grouped = False
                # for i in range(len(groups)):
                #     if axis == 'y':
                #         sample_y = groups[i][0][1]
                #         if abs(y1 - sample_y) < thresh:
                #             groups[i].append(line)
                #             grouped = True
                #     if axis == 'x':
                #         sample_x = groups[i][0][0]
                #         if abs(x1 - sample_x) < thresh:
                #             groups[i].append(line)
                #             grouped = True
                # if grouped is False:
                #     groups.append([line])



        poplist = []
        for i in range(len(groups)):
            if len(groups[i]) < group_num_thresh:
                poplist.append(i)

        self.remove_list_of_indexes(groups, poplist)


        return groups

    def check_axis(self, linelist=None):
        if linelist is None:
            linelist = self.linelist
        linelist = np.asarray(linelist)
        xvariance = np.std(linelist[:, 0])
        yvariance = np.std(linelist[:, 1])
        if xvariance < yvariance:
            axis = 'x'
        else:
            axis = 'y'
        return axis

    def find_balls(self, frame=None):
        if frame is None:
            frame = self.frame
        copy = frame.copy()
        cropbox = self.pocketbox
        top, bottom = cropbox[0][1], cropbox[1][1]
        left, right = cropbox[0][0], cropbox[1][0]

        copy[: top], copy[bottom: ] = (0, 0, 0), (0, 0, 0)
        copy[:, : left], copy[:, right:] = (0, 0, 0), (0, 0, 0)
        cv.imwrite('./debug_images/10_ball_area_crop.png', copy)

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

    def set_boxes(self):
        tablelines = self.tablelines
        pocketlines = self.pocketlines
        playlines = self.playfieldlines

        pocket_top_l = self.line_intersection(pocketlines['top'], pocketlines['left'])
        pocket_top_r = self.line_intersection(pocketlines['top'], pocketlines['right'])
        pocket_bot_l = self.line_intersection(pocketlines['bottom'], pocketlines['left'])
        pocket_bot_r = self.line_intersection(pocketlines['bottom'], pocketlines['right'])
        copy = self.frame.copy()
        cv.circle(copy, pocket_top_l, 5, (0, 0, 255), 4)
        cv.circle(copy, pocket_top_r, 5, (0, 0, 255), 4)
        cv.circle(copy, pocket_bot_l, 5, (0, 0, 255), 4)
        cv.circle(copy, pocket_bot_r, 5, (0, 0, 255), 4)
        cv.rectangle(copy, pocket_top_l, pocket_bot_r, (0, 255, 0), 2)
        cv.imwrite('./debug_images/9_pocket_box_check.png', copy)

        table_top_l = self.line_intersection(tablelines['top'], tablelines['left'])
        table_top_r = self.line_intersection(tablelines['top'], tablelines['right'])
        table_bot_l = self.line_intersection(tablelines['bottom'], tablelines['left'])
        table_bot_r = self.line_intersection(tablelines['bottom'], tablelines['right'])
        copy = self.frame.copy()
        cv.circle(copy, table_top_l, 5, (0, 0, 255), 4)
        cv.circle(copy, table_top_r, 5, (0, 0, 255), 4)
        cv.circle(copy, table_bot_l, 5, (0, 0, 255), 4)
        cv.circle(copy, table_bot_r, 5, (0, 0, 255), 4)
        cv.rectangle(copy, table_top_l, table_bot_r, (0, 255, 0), 2)
        cv.imwrite('./debug_images/9_table_box_check.png', copy)

        play_top_l = self.line_intersection(playlines['top'], playlines['left'])
        play_top_r = self.line_intersection(playlines['top'], playlines['right'])
        play_bot_l = self.line_intersection(playlines['bottom'], playlines['left'])
        play_bot_r = self.line_intersection(playlines['bottom'], playlines['right'])
        copy = self.frame.copy()
        cv.circle(copy, play_top_l, 5, (0, 0, 255), 4)
        cv.circle(copy, play_top_r, 5, (0, 0, 255), 4)
        cv.circle(copy, play_bot_l, 5, (0, 0, 255), 4)
        cv.circle(copy, play_bot_r, 5, (0, 0, 255), 4)
        cv.rectangle(copy, play_top_l, play_bot_r, (0, 255, 0), 2)
        cv.imwrite('./debug_images/9_playfield_box_check.png', copy)

        self.tablebox = (table_top_l, table_bot_r)
        self.pocketbox = (pocket_top_l, pocket_bot_r)
        self.playbox = (play_top_l, play_bot_r)

    @staticmethod
    def max_mid_min(group, axis='x'):
        if axis == 'x':
            xory = 0
        elif axis == 'y':
            xory = 1
        # print(len(group))
        group2 = list(group)
        maxind = group.argmax(axis=0)[xory]
        maximum = list(group2[maxind])
        group2.pop(maxind)
        group2 = np.asarray(group2)
        minind = group2.argmin(axis=0)[xory]
        minimum = list(group2[minind])
        group2 = list(group2)
        group2.pop(minind)
        midimum = list(group2[0])
        return maximum, midimum, minimum

    @staticmethod
    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
        sx1, sy1, sx2, sy2 = line2[0], line2[1], line2[2], line2[3]
        line1 = ((x1, y1), (x2, y2))
        line2 = ((sx1, sy1), (sx2, sy2))
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = int(det(d, xdiff) / div)
        y = int(det(d, ydiff) / div)
        pt = (x, y)
        return pt

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

    @staticmethod
    def remove_list_of_indexes(inlist, killlist):
        lencount = 0
        for i in range(len(killlist)):
            ind = killlist[i] - lencount
            lencount += 1
            del inlist[ind]
        return inlist

    # line_intersection((A, B), (C, D))

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

        shape = "square" if 1.05 >= ar >= 0.95 else "rectangle"


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
        table = Table(setting_num=setting)
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
