from imutils.video import FPS
import cv2 as cv
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import sqrt, atan
import cProfile
import pyglview
import time
from scipy.stats import zscore
from scipy.spatial.distance import cdist
import itertools
cv2 = cv

debugimages = False   # Debug makes playback much slower, but saves images of ball groups for each frame in frames folder
livegroups = True     # Displays the group circles live on playback
showgroupcolors = False
showtrajectory = True
n_wall_bounces = 3

slowdown = 0

ballcolors = [(230, 230, 225), (45, 25, 35), (160, 85, 50), (65, 85, 160), (60, 75, 223), (115, 190, 227), (86, 168, 225), (80, 110, 35), (148, 93, 228), (65, 30, 205)]

debuglist = []

filepath = './clips/2019_PoolChamp_Clip9.mp4'
unintruded = cv.imread('./cleanframe.png')

# Used to calculate the color difference of some potential ball to the table
table_color_bgr = (215, 145, 30)   # Green: (60, 105, 0)    Blue: (210, 140, 10)

# This is the number of balls in the frame at the start of the video. not yet used
nballs = 16
# This is the average radius in pixels of a ball in the images. There is a findballsize parameter in find_balls
# that can be used to identify average size
ballsize = 15

# This is the number of past frames to keep the found circles for identifying trajectories and deciding which
# balls are real
ballframebuffer = 10

# This setting is for the canny and lines used to identify the table boundaries
table_detect_setting = 2

# These are global for the shape of the frame, and the frame itself
shape = None
cur_frame = None
cleanframe = None
framenum = 0

# TODO: Use table color to determine whether a bumper line is misplaced. could also just use ratio
# TODO: Try analyzing ratio of h & w of table
# TODO: Put together a method for the Ball class that detects when it's in contact with another ball or the wall
# TODO: Take sample of shadow color from right inside the bumper box, and compare it to the color of a circle edge to /
#  see if it's just a shadow
# TODO: Create a Boundary or Rectangle class, and make subclasses for the playfield, pocket-area, and table-edges. the /
#  Table class is getting a bit big and unfocused.
# TODO: Have the colorthresh adapt to whether or not a ball is a blurry doublecircle


class Table:
    def __init__(self, setting_num=0):
        global shape
        self.edges = {}
        self.frame = None
        self.linelist = None
        self.playfieldlines = None
        self.pocketlines = None
        self.tablelines = None
        self.playbox = None
        self.pocketbox = None
        self.tablebox = None
        self.watchlist = []
        self.intersections = []
        self.setting = setting_num

        self.circles = []
        self.circlehistory = []
        self.potentialgroups = []
        self.balls = []
        self.allballs = []

        self.find_table_lines()
        self.find_all_intersections()
        self.goodpoints = self.check_all_intersections()
        self.check_line_ratios()
        self.set_boxes(self.goodpoints)
        self.pocketradius = int((self.playbox[1][0] - self.playbox[0][0]) / 48)
        self.pocketlocations = self.find_pockets()
        self.walls = self._get_walls()
        check = self.drawlines(color=(0, 200, 255))
        if debugimages:
            cv.imwrite('./debug_images/10_final_lines.png', check)
        print('Table Initialized!')

    def find_table_lines(self):
        global shape
        if cap.isOpened():
            ret, frame = cap.read()
            self.frame = frame.copy()
        count = 0
        n_frames = 10
        self.linelist = []

        print('Finding initial lines...')

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

            lines = cv2.HoughLinesP(canny, rho, np.pi / 180, 300, minLineLength=minlinelength, maxLineGap=maxlinegap)  # thresh was 300
            if lines is not None:
                for i in range(len(lines)):
                    self.linelist.append(lines[i][0])
            count += 1

        print(len(self.linelist), 'lines found')
        poplist = []

        print('Filtering lines by slope...')

        for i in range(len(self.linelist)):
            line = self.linelist[i]
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            if not (x2 - x1) == 0:  # makeing sure we don't get a divide by zero error
                slope = (y2 - y1) / (x2 - x1)  # rise over run
            if slope != 0:
                poplist.append(i)
        self.remove_list_of_indexes(self.linelist, poplist)
        print(len(poplist), 'lines removed')

        if debugimages:
            cv.imwrite('./debug_images/1_canny_check.png', canny)
            cannywlines = self.drawlines(canny)
            cv.imwrite('./debug_images/2_canny_check_lines.png', cannywlines)

            clean1 = self.drawlines(self.frame.copy(), color=(0, 0, 255))
            cv.imwrite('./debug_images/3_slope_filter.png', clean1)

        print('Sorting lines by direction...')

        horizontal, vertical = self.group_lines_by_direction(minlinelen=30)

        toplines, bottomlines = [], []
        leftlines, rightlines = [], []

        print('Sorting lines by side...')

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
        vert = self.drawlines(frame.copy(), linelist=vertical, color=(0, 0, 255))

        if debugimages:
            cv.imwrite('./debug_images/4_horizontal_split.png', horiz)
            cv.imwrite('./debug_images/4_vertical_split.png', vert)

        self.linelist = {'top': toplines, 'bottom': bottomlines, 'left': leftlines, 'right': rightlines}
        print('Grouping lines by proximity and replacing them with an averaged line...')

        for key in self.linelist:
            lines = self.linelist[key]
            grouped = self.group_lines_by_proximity(lines)

            for i in range(len(grouped)):
                checkimg = self.drawlines(frame.copy(), linelist=grouped[i], color=(0, 0, 255))
                cv.imwrite('./debug_images/5_' + key + str(i) + '_lines.png', checkimg)

            if key == 'top' or key == 'bottom':
                axis = 'y'
            else:
                axis = 'x'
            averaged = self.find_average_lines_from_groups(grouped, axis)
            self.linelist[key] = averaged

            if debugimages:
                checkimg = self.drawlines(frame.copy(), linelist=grouped, color=(0, 0, 255))
                cv.imwrite('./debug_images/5_' + key + 'lines.png', checkimg)
                checkimg = self.drawlines(frame.copy(), linelist=averaged, color=(0, 0, 255))
                cv.imwrite('./debug_images/6_averaged_' + key + 'lines.png', checkimg)
                copy2 = self.drawlines(frame.copy(), color=(0, 0, 255))
                cv.imwrite('./debug_images/7_grouped.png', copy2)

        print('Lines Found')

        return frame

    def find_pockets(self):
        pockets = self.goodpoints[0]
        tablemid = int(self.playbox[0][0] + ((self.playbox[1][0] - self.playbox[0][0]) // 2))
        topmid = (tablemid, self.playbox[0][1])
        botmid = (tablemid, self.playbox[1][1])
        pockets.append(topmid)
        pockets.append(botmid)

        if debugimages:
            copy = self.frame.copy()
            for i in pockets:
                cv.circle(copy, i, self.pocketradius, (255, 0, 200), 2)
            cv.imwrite('./debug_images/20_pocket_locations.png', copy)
        print('pocketsdrawn:', pockets)
        return pockets

    def drawlines(self, frame=None, linelist=None, color=(0, 0, 255), thickness=2):

        def draw(line_list):
            for lines_ in line_list:
                if isinstance(lines_[0], (int, float, np.int32, np.int64)):
                    x1, y1, x2, y2 = lines_[0], lines_[1], lines_[2], lines_[3]
                    pt1, pt2 = (x1, y1), (x2, y2)
                    cv.line(frame, pt1, pt2, color, thickness)
                else:
                    for line in lines_:
                        if isinstance(line[0], (int, float, np.int32, np.int64)):
                            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
                            pt1, pt2 = (x1, y1), (x2, y2)
                            cv.line(frame, pt1, pt2, color, thickness)

        if frame is None:
            frame = self.frame.copy()
        if len(frame.shape) == 2:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        if linelist is None:
            linelist = self.linelist

        if isinstance(linelist, (list, np.ndarray)):
            draw(linelist)
        elif isinstance(linelist, dict):
            for key in linelist:
                lines = linelist[key]
                draw([lines])
        return frame

    def drawboxes(self, frame=None, boxes=None):
        if frame is None:
            frame = self.frame.copy()
        if boxes is None:
            boxes = [self.playbox, self.pocketbox, self.tablebox]
        for box in boxes:
            pt1, pt2 = box[0], box[1]
            cv.rectangle(frame, pt1, pt2, (255, 255, 255), 2)

    def recenter_coordinates(self, coords=(0, 0)):
        frame = self.frame.copy()
        height = frame.shape[0]
        width = frame.shape[1]
        if isinstance(coords[0], (int, float, np.int32, np.int64)):
            x = coords[0] + (width // 2)
            y = coords[1] + (height // 2)
            outcoords = (x, y)
            return outcoords
        elif isinstance(coords[0], (list, tuple, np.ndarray)):
            outcoordlist = []
            for l in coords:
                x = coords[0] + (width // 2)
                y = coords[1] + (height // 2)
                outcoordlist.append((x, y))
            return outcoordlist

    def check_line_ratios(self, linedict=None):

        play = self.goodpoints[0]
        pocket = self.goodpoints[1]
        table = self.goodpoints[2]

        playlong = play[1][0] - play[0][0]
        playshort = play[2][1] - play[1][1]
        playratio = playlong / playshort

        pocketlong = pocket[1][0] - pocket[0][0]
        pocketshort = pocket[2][1] - pocket[1][1]
        pocketratio = pocketlong / pocketshort

        tablelong = table[1][0] - table[0][0]
        tableshort = table[2][1] - table[1][1]
        tableratio = tablelong / tableshort

        print('ratios:', playratio, pocketratio, tableratio)

        # print('Checking line distances by part of table...')
        #
        # def get_relative_distances():
        #     distances = {'full': [], 'wood': [], 'bump': []}
        #     for key_ in goodkeys:
        #         if key_ == 'top':
        #             ind = 1
        #             top = linedict[key_]
        #             playfield, pockets, table = self.max_mid_min(top, axis='y')
        #         elif key_ == 'bottom':
        #             ind = 1
        #             bottom = linedict[key_]
        #             table, pockets, playfield = self.max_mid_min(bottom, axis='y')
        #         elif key_ == 'left':
        #             ind = 0
        #             left = linedict[key_]
        #             playfield, pockets, table = self.max_mid_min(left, axis='x')
        #         elif key_ == 'right':
        #             ind = 0
        #             right = linedict[key_]
        #             table, pockets, playfield = self.max_mid_min(right, axis='x')
        #         else:
        #             print('line_dict is not in the right format in check_line_ratios')
        #         loc_play = playfield[ind]
        #         loc_poc = pockets[ind]
        #         loc_tab = table[ind]
        #
        #         dist_tab_play = abs(loc_tab - loc_play)
        #         dist_tab_poc = abs(loc_tab - loc_poc)
        #         dist_poc_play = abs(loc_poc - loc_play)
        #
        #         distances['full'].append(dist_tab_play)
        #         distances['wood'].append(dist_tab_poc)
        #         distances['bump'].append(dist_poc_play)
        #     avg_widths = {'full': [], 'wood': [], 'bump': []}
        #     for key_ in distances:
        #         dists = distances[key_]
        #         dev = np.std(dists)
        #         if dev > 7:  # TODO: may need to tune deviation
        #             print('\ndeviaition is high in ' + key_)
        #             print('you might wanna check the debug images.\n')
        #         else:
        #             avg = sum(dists) / len(dists)
        #             avg_widths[key_].append(int(avg))
        #     return avg_widths
        #
        # def check_against_avgs(checklines, xory, avg_dict, thresh=10):
        #     checklines = [list(i) for i in checklines]
        #     dist_list = [val[0] for val in avg_dict.values()]
        #     keeplines = []
        #     for i in range(len(checklines)):
        #         line = checklines[i]
        #         for j in range(i + 1, len(checklines)):                      # TODO: KEEP AND EYE OUT HERE FOR ISSUES
        #             vs_line = checklines[j]
        #             dist = abs(line[xory] - vs_line[xory])
        #             for i in range(dist - thresh, dist + thresh):
        #                 if int(i) in dist_list:
        #                     if line not in keeplines:
        #                         keeplines.append(line)
        #                     if vs_line not in keeplines:
        #                         keeplines.append(vs_line)
        #
        # if linedict is None:
        #     linedict = self.linelist
        # goodkeys = []
        # badkeys = []
        # for key in linedict:
        #     group = linedict[key]
        #     if len(group) == 3:
        #         goodkeys.append(key)
        #     else:
        #         badkeys.append(key)
        #
        # for key in badkeys:
        #     group = linedict[key]
        #     if len(group) < 3:
        #         print('Not enough lines found for the ' + key + ' side. You may need to change the setting or tune some'
        #               ' parameters in Table.find_table_lines()')
        #     if len(group) > 3:
        #         if key == 'top' or key == 'bottom':
        #             axis = 1
        #         elif key == 'left' or key == 'right':
        #             axis = 0
        #         average_dists = get_relative_distances()
        #         check_against_avgs(group, axis, average_dists)

    def group_lines_by_direction(self, minlinelen=0):
        vertical = []
        horizontal = []
        for i in range(len(self.linelist)):
            line = self.linelist[i]
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            if x1 == x2 and abs(y1 - y2) > minlinelen:
                vertical.append(self.linelist[i])
            elif y1 == y2 and abs(x1 - x2) > minlinelen:
                horizontal.append(self.linelist[i])
        return horizontal, vertical

    def group_lines_by_proximity(self, linelist=None, thresh=20, group_num_thresh=3):

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

        groups = []
        axis = self.check_axis(linelist)
        if linelist is None:
            linelist = self.linelist
        if isinstance(linelist, dict):
            for key in linelist:
                list_ = linelist[key]
        elif isinstance(linelist, (list, np.ndarray)):
            for line in linelist:
                group(line)
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

    def draw_circles(self, circles, frame=None):
        if frame is None:
            frame = self.frame.copy()

        # print('circles', type(circles))
        if isinstance(circles[0], Circle):
            circles = [(i.center[0], i.center[1], i.radius) for i in circles]
            circles = np.uint16(np.around(circles))
        else:
            circles = np.uint16(np.around(circles))
        for circle in circles:
            x, y = circle[0], circle[1]
            radius = circle[2]
            cv.circle(frame, (x, y), radius, (0, 255, 0), 1)
            cv.circle(frame, (x, y), 2, (0, 0, 255), 2)
            rect = self.calculate_radius_square(radius, (x, y))
            cv.rectangle(frame, rect[0], rect[1], (255, 0, 255), 1)
        return frame

    def set_boxes(self, goodpoints):
        playfield_points = goodpoints[0]
        pocket_points = goodpoints[1]
        table_points = goodpoints[2]
        for pt in playfield_points:
            quad = self.check_quadrant(pt)
            if quad == 'ul':
                ul = pt
            elif quad == 'br':
                br = pt
        self.playbox = [ul, br]

        for pt in pocket_points:
            quad = self.check_quadrant(pt)
            if quad == 'ul':
                ul = pt
            elif quad == 'br':
                br = pt
        self.pocketbox = [ul, br]

        for pt in table_points:
            quad = self.check_quadrant(pt)
            if quad == 'ul':
                ul = pt
            elif quad == 'br':
                br = pt
        self.tablebox = [ul, br]
        if debugimages:
            copy = self.frame.copy()
            cv.rectangle(copy, self.playbox[0], self.playbox[1], (0, 200, 255), 2)
            cv.imwrite('./debug_images/8_playbox_check.png', copy)
            copy = self.frame.copy()
            cv.rectangle(copy, self.pocketbox[0], self.pocketbox[1], (0, 200, 255), 2)
            cv.imwrite('./debug_images/8_pocketbox_check.png', copy)
            copy = self.frame.copy()
            cv.rectangle(copy, self.tablebox[0], self.tablebox[1], (0, 200, 255), 2)
            cv.imwrite('./debug_images/8_tablebox_check.png', copy)

    def find_all_intersections(self, lines=None):
        print('Finding all intersections...')
        if lines is None:
            lines = self.linelist
        if isinstance(lines, dict):
            lines = list(lines.values())
        all_lines = []
        if isinstance(lines[0], list) and not isinstance(lines[0][0], (int, float, np.int32, np.int64)):
            for linelist in lines:
                if isinstance(linelist[0], list) and isinstance(linelist[0][0], (int, float, np.int32, np.int64)):
                    for lst in lines:
                        for line in lst:
                            all_lines.append(line)
                else:
                    print('Wrong type in Table.find_all_intersections')
        else:
            print('Wrong type in Table.find_all_intersections')
        for line in all_lines:
            for i in range(len(all_lines)):
                line2 = all_lines[i]
                if line != line2 and line is not None and line2 is not None:
                    intersection = self.find_line_intersection(line, line2)
                    if intersection not in self.intersections and intersection is not None:
                        self.intersections.append(intersection)

    def check_all_intersections(self):

        midx = self.frame.shape[1] // 2
        midy = self.frame.shape[0] // 2

        def get_dists_from_center(points):

            shifted = [(xy[0] - midx, xy[1] - midy) for xy in points]  # shifting all coordinates so their distance from center can be easily detirmined
            shifted.sort(key=lambda x: abs(x[0]))  # sorting the shifted list by the absolute value of x
            coords_w_dists = []
            all_x_dists = []
            all_y_dists = []
            for i in range(len(shifted)):
                coord = shifted[i]

                centerdist, xdist, ydist = self.get_dist(coord, (0, 0), seperate=True)
                all_x_dists.append(xdist)
                all_y_dists.append(ydist)
                coord = (coord[0] + midx, coord[1] + midy)
                coord_dist = (coord, centerdist)
                coords_w_dists.append(coord_dist)
            coords_w_dists.sort(key=lambda x: x[0])
            coords_w_dists.sort(key=lambda x: x[1])

            return coords_w_dists

        def check_quadrant(point):
            out = ''
            x, y = point[0], point[1]
            if y < midy:
                out += 'u'
            elif y > midy:
                out += 'b'
            if x < midx:
                out += 'l'
            elif x > midx:
                out += 'r'
            return out

        def find_good_points(p_w_d):
            quadrants = {'ul': [], 'ur': [], 'br': [], 'bl': []}
            for ptdist in p_w_d:
                point = ptdist[0]
                key = self.check_quadrant(point)
                quadrants[key].append(ptdist)
            for key in quadrants:
                quadrants[key].sort(key=lambda x: x[1])
                quadrants[key] = [i[0] for i in quadrants[key]]
            grouplist = []
            goodpoints = {0: [], 1: [], 2: []}
            for i in range(3):                                               # For each or the 3 parts
                group = []                                                      # Create a new group of points
                copy = self.frame.copy()
                for key in quadrants:                                           # For each group in quadrants
                    pt = quadrants[key][0]

                    group.append(pt)                                                # add the smallest of the quadrant to group
                    cv.circle(copy, pt, 5, (255, 255, 255), 5)
                    del quadrants[key][0]                                           # delete the value that you just added from the quadrant

                grouplist.append(group)                                         # add the group you just made to the list of groups
                goodpoints[i] = group
                for key in quadrants:                                           # For each group in quadrants:
                    quadrant = quadrants[key]
                    killlist = []
                    for ind in range(len(quadrant)):                                # For each index in the current quadrant
                        point = quadrant[ind]
                        for group in grouplist:                                         # For each group in the list of already-added groups
                            for gpoint in group:                                            # For each known good point in that group
                                if ind not in killlist and isinline(gpoint, point):                                     # if the point in the current quadrant at the current index is inline with the good point
                                    killlist.append(ind)                                            # Add that index to the killlist
                    quadrants[key] = self.remove_list_of_indexes(quadrants[key], killlist)   # remove all indexes in the killlist from the quadrant

                if debugimages:
                    cv.imwrite('./debug_images/14_quadrant_minimum_point_check' + str(i) + '.png', copy)
            return goodpoints

        def check_ratio(dist1, dist2, thresh=0.2):
            dists = [dist1, dist2]
            shorter = min(dists)
            longer = max(dists)
            longratio = 2 - (longer / shorter)
            if abs(longratio) < thresh:
                print('hell yea, ratio be good.')
                return True
            else:
                print('ratio was ' + str(longratio - thresh) + ' off from the threshold')
                return False

        def isinline(pt1, pt2, return_axis=False):
            x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
            if x1 == x2 and y1 == y2:
                print('isinline is being given two of the same point')
                if return_axis:
                    return True, 'xy'
                else:
                    return True
            elif x1 == x2:
                if return_axis:
                    return True, 'x'
                else:
                    return True
            elif y1 == y2:
                if return_axis:
                    return True, 'y'
                else:
                    return True
            else:
                return False

        intersections = self.intersections
        points_w_dists = get_dists_from_center(intersections)
        good_points = find_good_points(points_w_dists)
        return good_points

    def check_quadrant(self, point):
        midx = self.frame.shape[1] // 2
        midy = self.frame.shape[0] // 2
        out = ''
        x, y = point[0], point[1]
        if y < midy:
            out += 'u'
        elif y > midy:
            out += 'b'
        if x < midx:
            out += 'l'
        elif x > midx:
            out += 'r'
        return out

    def find_balls(self, frame=None, draw=True, setting=3, findballsize=False):
        colorthresh1 = 20
        colorthresh2 = 100

        if frame is None:
            frame = cur_frame
        copy = frame.copy()
        cropbox = self.pocketbox
        top, bottom = cropbox[0][1], cropbox[1][1]
        left, right = cropbox[0][0], cropbox[1][0]

        copy = copy[top:bottom, left:right]

        if debugimages:
            cv.imwrite('./debug_images/10_ball_area_crop.png', copy)

        gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (3, 3), 5)

        if setting == 1:
            min_dist = frame.shape[0] / 64
            max_radius = round(frame.shape[0] / 22.5)  # was / 22.5
            minradius = 5
            param1 = 53
            param2 = 30
            dp = 1

        elif setting == 2:
            min_dist = 10
            ballsize_thresh = 1
            max_radius = ballsize + ballsize_thresh
            minradius = ballsize - ballsize_thresh
            param1 = 80
            param2 = 20
            dp = 1.5

        elif setting == 3:
            min_dist = 10
            ballsize_thresh = 1
            max_radius = ballsize + ballsize_thresh
            minradius = ballsize - ballsize_thresh
            param1 = 60
            param2 = 27
            dp = 1.9

        if findballsize:
            minradius = 0
            max_radius = 50
            draw = True

        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=minradius, maxRadius=max_radius)

        if circles is not None:
            circles = [(int(i[0] + left), int(i[1] + top), i[2]) for i in circles[0]]

            self.add_circles_to_log(circles)

            for circle in self.circles:

                if circle.inpocket:
                    thresh = colorthresh2
                else:
                    thresh = colorthresh1
                bcoldifflist = [color_diff(circle.color, i) for i in ballcolors]
                # print(min(bcoldifflist))
                if min(bcoldifflist) > 80:
                    self.circles.remove(circle)
            if draw:
                frame = self.draw_circles(self.circles, frame)
            if findballsize:
                self.get_ball_size(circles)

            self.average_motionblur_circles()
            # self.hsv_test()
            # if framenum % 10 == 0:
            #     self.detect_intrusion()
            self.find_ball_groups()
        return frame

    def add_circles_to_log(self, circles):
        if len(self.circles) > 0:
            self.circlehistory.append(self.circles)
            self.circles = []
        if len(self.circlehistory) > ballframebuffer:
            del self.circlehistory[0]

        for ind in range(len(circles)):
            circ = circles[ind]
            x, y, r = circ[0], circ[1], circ[2]
            potentialball = Circle((x, y), r)
            # newball.findlastposition(self.ballhistory)

            # if potentialball.line is not None:
            #     pt1, pt2 = potentialball.line[0], potentialball.line[1]
            #     cv.line(frame, pt1, pt2, (255, 255, 255), 2)
            self.circles.append(potentialball)

    def find_ball_groups(self):
        history = self.circlehistory
        balls = self.balls
        distthresh = 110
        regcolorthresh = 90
        blurcolorthresh = 200
        tablecolorthresh = 80
        # if len(history) > 0:
        circles = self.circles
        # At the start of the program, start a new group

        if len(balls) == 0:
            if len(history) > 0:
                oldcircles = history[-1]
                for circ in circles:
                    circs_w_col_n_dists = self.colordiffs_n_distances(circ, oldcircles)
                    if circs_w_col_n_dists[0][1] < distthresh:
                        closest = circs_w_col_n_dists[0][0]
                        newball = Ball(circ)
                        newball.append(closest)
                        newball.append(circ)
                        self.balls.append(newball)
                        self.allballs.append(newball)

        # Once there's a place to start for the group comparisons:
        else:
            oldcircles = [i[-1] for i in balls]
            ungrouped = []
            # circlevecs = np.array([circ.color for circ in circles])
            # oldcirclevecs = np.array([circ.color for circ in oldcircles])
            #
            # distmap = cdist(oldcirclevecs, circlevecs)
            # distmap = np.round(distmap, 1)
            #
            # allinds = []
            # for row in distmap:
            #     inds = np.argsort(row)[:3]
            #     allinds.append(inds)
            #
            # allinds = np.array(allinds)
            # first = allinds[:, 0]
            #
            # # mins = np.argmin(distmap, axis=0)
            # unique = len(np.unique(first))
            # if unique < len(first):
            #     print('multiple circles matched to ball')
            # #
            # # print('distvecs', distmap)
            # # print('argmiin:', mins)
            for circ in circles:
                # badcirc = False
                # print('self.watchlist', self.watchlist)
                # for ball in self.watchlist:
                #     color = ball.color
                #     dist = distance(ball.center, circ.center)
                #     watchlistcoldiff = color_diff(circ.color, color)
                #     print('watchlistdiff:', watchlistcoldiff)
                #     if watchlistcoldiff < 40 and dist < 50:
                #         # self.watchlist.append(circ.color)
                #         badcirc = True  #TODO: Append entire ball, rather than just color, and use location to detirmine
                #                          #TODO: Also make the items in watchlist clearthemselves after a while.
                #         break
                # if badcirc:
                #     continue

                if circ.isblurred or circ.iswhite:
                    colorthresh = blurcolorthresh
                else:
                    colorthresh = regcolorthresh
                grouped = False
                circs_w_col_n_dists = self.colordiffs_n_distances(circ, oldcircles)
                for j in range(len(circs_w_col_n_dists)):
                    closest = circs_w_col_n_dists[j][0]
                    distfrom = circs_w_col_n_dists[j][1]
                    colordiff = circ.compare_color(closest)
                    tablediff = color_diff(circ.color, table_color_bgr)
                    if j == 0:
                        debuginfo = {'frame': framenum, 'center': circ.center,'tablediff': tablediff, 'color': circ.color, 'closestcol': closest.color,'colordiff': colordiff, 'closestdist': distfrom}
                    if tablediff < tablecolorthresh or circ.ispartial:
                        print(str(framenum), 'too close to table color')
                        break
                    elif colordiff < colorthresh:
                        for i in range(len(balls)):
                            ball = balls[i]
                            if distfrom < distthresh * (2 * ball.nmissingframes + 1):
                                if closest in ball:
                                    # if group.inpocket and circ.is_past_boundary(self.playbox)
                                    if not ball.inpocket:
                                        ball.append(circ)
                                        grouped = True
                                    elif circ.is_past_boundary(self.playbox):
                                        ball.append(circ)
                                        grouped = True
                                    if ball.is_past_boundary(self.playbox):
                                        ball.inpocket = True
                                    break
                        if grouped:
                            break
                if not grouped:
                    debuglist.append(debuginfo)
                    if not circ.is_past_boundary(self.playbox) and not circ.ispartial:
                        potential_ball = Ball(circ)
                        potential_ball.append(circ)
                        balls.append(potential_ball)
                        self.allballs.append(potential_ball)
                    print('frame:', framenum)
                    print(circ, circ.color, 'not grouped. was looking for', closest, closest.color)
                    print('distfrom:', distfrom, 'colordiff:', colordiff)
                    ungrouped.append(circ)

            for ball in balls:
                ball.update_lastseen()
                # if ball.variance:
                #     if ball.variance > 0.25:
                #         print('ball', ball.color, 'removed for variance of:', ball.variance)
                #         balls.remove(ball)
                if ball.nmissingframes > 5 and len(ball) < 3:
                    balls.remove(ball)
                elif ball.nmissingframes > 20:
                    balls.remove(ball)

            colors = [ball.color for ball in balls]
            # colorshsv = [ball.hsv for ball in groups]

            if debugimages or livegroups:
                if debugimages and not livegroups:
                    copy = cur_frame.copy()
                else:
                    copy = cur_frame
                for i in range(len(balls)):
                    ball = balls[i]
                    # if group.inpocket:
                    #     color = (0, 255, 0)
                    # else:
                    if ball.collisioncourse:
                        color = (0, 0, 255)
                    else:
                        color = colors[i]

                    # hsvcol = colorshsv[i]
                    for c in ball:
                        cv.circle(copy, c.center, c.radius, color, 2)
                    if showgroupcolors:
                        cv.putText(copy, str(ball.color), (shape[1] - 600, 300 + (40 * i)), cv.FONT_HERSHEY_PLAIN, 2, color, thickness=4)
                        cv.putText(copy, str(ball.variance), (shape[1] - 300, 300 + (40 * i)), cv.FONT_HERSHEY_PLAIN, 2, color, thickness=4)
                        # cv.putText(copy, str(ball.hsv), (shape[1] - 300, 300 + (40 * i)), cv.FONT_HERSHEY_PLAIN, 2, hsvcol, thickness=4)
                for c in ungrouped:
                    cv.circle(copy, c.center, c.radius, (0, 0, 0), 2)
                cv.putText(copy, 'frame: ' + str(framenum), (20, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
                cv.putText(copy, str(len(balls)) + ' groups', (20, shape[0] - 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
                if debugimages:
                    cv.imwrite('./debug_images/frames/16_ungrouped_circles_' + str(framenum) + '.png', copy)

    def average_motionblur_circles(self, tolerance=2, colorthresh=90):
        killlist = []
        ilist = []
        for i in range(len(self.circles)):
            ilist.append(i)
            circle = self.circles[i]
            for j in range(len(self.circles)):
                circle2 = self.circles[j]
                if j in ilist:
                    continue
                else:
                    colordiff = circle.compare_color(circle2)
                    dist = distance(circle.center, circle2.center)
                    joined_radius = circle.radius + circle2.radius
                    if dist < joined_radius - tolerance:
                        if colordiff < colorthresh:
                            print('doublecircle colordiff', colordiff)
                            print(framenum, 'double circle found!!')
                            print('dist:', dist, 'joined radius:', joined_radius, 'color difference:', colordiff)
                            avgrad = joined_radius / 2
                            avgpoint = self.halfway_point(circle.center, circle2.center)
                            newcirc = Circle(avgpoint, int(avgrad))
                            self.circles[i] = newcirc
                            killlist.append(j)
        self.remove_list_of_indexes(self.circles, killlist)

    # def detect_intrusion(self):
    #
    #     outer = self.pocketbox
    #     inner = self.playbox
    #
    #     otop, obot = outer[0][1], outer[1][1]
    #     oleft, oright = outer[0][0], outer[1][0]
    #     itop, ibot = inner[0][1], inner[1][1]
    #     ileft, iright = inner[0][0], inner[1][0]
    #     copy = cleanframe
    #     top = copy[otop: itop, ileft: iright]
    #     bot = copy[ibot: obot, ileft: iright]
    #     left = copy[itop: ibot, oleft: ileft]
    #     right = copy[itop: ibot, iright: oright]
    #
    #     z = [0, 0, 0]
    #
    #     # topmean = np.mean(top, axis=0)
    #     # toprows = [[255, 255, 255] if color_diff(i, table_color_bgr) > 160 else [z] for i in topmean]
    #     # toprows = np.asarray(toprows)
    #     #
    #     # botmean = np.mean(bot, axis=0)
    #     # botrows = [[255, 255, 255] if color_diff(i, table_color_bgr) > 160 else [z] for i in botmean]
    #     # botrows = np.asarray(botrows)
    #
    #     lmean = np.mean(left, axis=1)
    #     tcolor = np.asarray(table_color_bgr)
    #     diffs = abs(lmean - tcolor)
    #
    #
    #     # leftrows = [[255, 255, 255] if color_diff(i, table_color_bgr) > 160 else [z, z, z] for i in leftmean]
    #
    #     # leftrows = np.asarray(leftrows)
    #     # leftrows = cv.cvtColor(leftrows, cv.COLOR_BGR2GRAY)
    #     # retval, leftrows = cv.threshold(leftrows, 50, 255, cv.THRESH_BINARY)
    #
    #     # rightmean = np.mean(right, axis=1)
    #     # rightrows = [[255, 255, 255] if color_diff(i, table_color_bgr) > 160 else [z] for i in rightmean]
    #     # rightrows = np.asarray(rightrows)
    #
    #
    #
    #     # cv.imwrite('./debug_images/20_topmean.png', toprows)
    #     # cv.imwrite('./debug_images/20_botmean.png', botrows)
    #     cv.imwrite('./debug_images/20_leftmean.png', leftrows)
    #     # cv.imwrite('./debug_images/20_rightmean.png', rightrows)

    @staticmethod
    def halfway_point(pt1, pt2):
        x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
        avgx = int((x1 + x2) / 2)
        avgy = int((y1 + y2) / 2)
        newpoint = (avgx, avgy)
        return newpoint

    @staticmethod
    def calculate_radius_square(radius, center):
        x1, y1 = center[0], center[1]
        r = radius
        sqr_length = (r ** 2) / 2
        dist = sqrt(sqr_length)
        tl = (int(x1 - dist), int(y1 - dist))
        br = (int(x1 + dist), int(y1 + dist))
        return tl, br

    @staticmethod
    def get_ball_size(circles):
        radiuslist = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y = circle[0], circle[1]
                radius = circle[2]
                radiuslist.append(radius)
                # if draw:
                #     cv.circle(frame, (x, y), radius, (0, 255, 0), 1)
                #     cv.circle(frame, (x, y), 2, (0, 0, 255), 2)
            print('circle radius max:', str(max(radiuslist)), 'min:', str(min(radiuslist)), 'avg:', str(sum(radiuslist) / len(radiuslist)))

    @staticmethod
    def colordiffs_n_distances(circ, oldcircles):
        circs_w_dists = [
            (i, distance(circ.center, i.center),
             color_diff(circ.color, i.color)) for i in oldcircles
        ]
        circs_w_dists.sort(key=lambda x: x[2])
        return circs_w_dists

    # @staticmethod
    # def color_diff_list(circ, oldcircles):
    #     circs_w_colordiffs = []
    #     for ocirc in oldcircles:
    #         diff = color_diff(circ.color, ocirc.color)
    #         circs_w_colordiffs.append((ocirc, diff))
    #     circs_w_colordiffs.sort(key=lambda x: x[1])
    #     print(circs_w_colordiffs)
    #     return circs_w_colordiffs

    @staticmethod
    def max_mid_min(group, axis='x'):
        if axis == 'x':
            xory = 0
        elif axis == 'y':
            xory = 1
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
    def find_line_intersection(line1, line2):
        x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
        sx1, sy1, sx2, sy2 = line2[0], line2[1], line2[2], line2[3]
        line1 = ((x1, y1), (x2, y2))
        line2 = ((sx1, sy1), (sx2, sy2))
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)

        if div != 0:
            d = (det(*line1), det(*line2))
            x = int(det(d, xdiff) / div)
            y = int(det(d, ydiff) / div)
            pt = (x, y)
            return pt

    @staticmethod
    def find_average_lines_from_groups(groups, axis='x'):
        output = []
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

    @staticmethod
    def get_dist(pt1, pt2, seperate=False):
        x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
        xdist = abs(x1 - x2)
        ydist = abs(y1 - y2)
        if xdist == 0:
            dist = ydist
        elif ydist == 0:
            dist = xdist
        else:
            dist = sqrt((xdist ** 2) + (ydist ** 2))
        if seperate is True:
            return dist, xdist, ydist
        else:
            return dist

    def _get_walls(self):
        boxpoints = self.goodpoints[0]
        return {'top': (boxpoints[0], boxpoints[1]), 'bottom': (boxpoints[2], boxpoints[3]), 'left': (boxpoints[0], boxpoints[3]), 'right': (boxpoints[1], boxpoints[2])}
        # def _
        # walls = {side: lambda x: ta
        #     upper_bound = np.array((table.playbox[1][0], table.playbox[0][1]))
        # lower_bound = np.array((table.playbox[1][0], table.playbox[1][1]))
        # }


class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.isblurred = False
        self.iswhite = False
        self.ispartial = False
        self.color = self.get_ball_color()
        self.inpocket = self.is_past_boundary(table.playbox)
        if len([*filter(lambda x: x >= 160, self.color)]) == 3:
            self.iswhite = True
        # if all(self.color) > 160:
        #     self.iswhite = True
        # self.hsv = self.get_ball_color(hsv=True)

    def __repr__(self):
        return str(self.center)

    def calculate_radius_square(self):
        x1, y1 = self.center[0], self.center[1]
        r = self.radius
        sqr_length = (r ** 2) / 2
        dist = sqrt(sqr_length)
        tl = (int(x1 - dist), int(y1 - dist))
        br = (int(x1 + dist), int(y1 + dist))
        return tl, br

    def get_ball_color(self, hsv=False):
        tl, br = self.calculate_radius_square()
        x1, y1 = tl[0], tl[1]
        x2, y2 = br[0], br[1]
        cropped = cleanframe[y1: y2, x1: x2]
        if hsv:
            cropped = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
        if debugimages:
            cv.imwrite('./debug_images/ball_crop/15_cropped_ball.png', cropped)
        avg_b = int(np.mean(cropped[:, :, 0]))
        avg_g = int(np.mean(cropped[:, :, 1]))
        avg_r = int(np.mean(cropped[:, :, 2]))
        avg = (avg_b, avg_g, avg_r)

        rowmean = np.mean(cropped, axis=0)
        colmean = np.mean(cropped, axis=1)

        closerowdiffs = [i for i in rowmean if color_diff(i, table_color_bgr) < 70]
        closecoldiffs = [i for i in colmean if color_diff(i, table_color_bgr) < 70]
        # for i in rowmean:
        #     diff = color_diff(i, table_color_bgr)
        #     if diff < 70:
        #         closediffs.append(diff)
        combined = len(closerowdiffs) + len(closecoldiffs)
        if combined > 15:
            print('table circle found')
            self.ispartial = True
        elif combined > 3:
            self.ispartial = True
            print('partial circle found')


        # print(avg)
        return avg

    def compare_color(self, circ2, hsv=False):
        if hsv:
            hsv1 = self.hsv
            hsv2 = circ2.hsv
            diff = color_diff(hsv1, hsv2)
            return diff
        else:
            col1 = self.color
            col2 = circ2.color
            diff = color_diff(col1, col2)
            return diff

    def is_past_boundary(self, box):
        x, y = self.center[0], self.center[1]
        xmin, ymin = box[0][0], box[0][1]
        xmax, ymax = box[1][0], box[1][1]
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return True
        else:
            return False


class Ball(Circle):
    def __init__(self, circle):
        super().__init__(circle.center, circle.radius)
        self.movementthresh = 5
        self.past = Buffer(fullhist=False)
        self.colorpast = Buffer(fullhist=False, maxlength=ballframebuffer * 3)
        self.latestcolor = None
        self.queball = False
        self.eightball = False
        self.ismoving = False
        self.motion = Motion
        self.lastseenframe = None
        self.coefficients = None
        self.colliding = None
        self.collisioncourse = False
        self.velocitypast = Buffer(fullhist=False)
        self.speed = 0
        self.motionlines = []
        self.speedlist = []
        self.variancebuffer = []
        self.variancehistory = np.array([])
        self.variance = 0
        self.nmissingframes = 0
        self.watchlist = []

    def __str__(self):
        return str(self.past)

    def __iter__(self):
        return iter(self.past)

    def __getitem__(self, i):
        return self.past[i]

    def __setitem__(self, i, value):
        self.past[i] = value

    def __delitem__(self, i):
        del self.past[i]

    def __len__(self):
        return len(self.past)

    def __index__(self, i):
        return self.past[i]

    def append(self, item):
        self.past.append(item)

        self.lastseenframe = framenum
        self.center = item.center

        self.colorpast.append(item.color)

        if len(self.colorpast) >= ballframebuffer:
            self.set_avgcolor()

        recent = self.past[-5:]
        distpast = []
        for i in range(len(recent) - 1):
            pt1, pt2 = recent[i].center, recent[i+1].center
            dist = distance(pt1, pt2)
            distpast.append(dist)

        # if len([*filter(lambda x: x >= self.movementthresh, distpast)]) > 0:
        if len(distpast) > 2:
            self.speed = sum(distpast) / len(distpast)
            self.speedlist.append(self.speed)
            if len(self.speedlist) > ballframebuffer:
                del self.speedlist[0]
            self.ismoving = len([*filter(lambda x: x >= self.movementthresh, distpast)]) > 0 and self.speed > self.movementthresh
            if self.ismoving:
                if showtrajectory:
                    if len(self.past) >= ballframebuffer and self.nmissingframes < 3 and not self.inpocket:
                        self.find_future_path()
            else:
                self.velocitypast.append((0, 0))
                # self.velocitypast = np.append(self.velocitypast, (0, 0))
        else:
            self.ismoving = False
            self.variancebuffer.append((0, 0))
        self.ischaotic()

    def update_lastseen(self):
        self.nmissingframes = framenum - self.lastseenframe

    def set_avgcolor(self):
        pastcolors = np.asarray(self.colorpast)
        length = len(pastcolors)
        b = int(sum(pastcolors[:, 0]) / length)
        g = int(sum(pastcolors[:, 1]) / length)
        r = int(sum(pastcolors[:, 2]) / length)
        self.color = (b, g, r)

    def find_future_path(self):
        self.motionlines = []
        velocity_multiplier = 60

        speed = sum(self.speedlist) / len(self.speedlist)
        vel = speed * velocity_multiplier
        self.motion.speed = vel

        past = np.asarray([i.center for i in self.past])

        ballpt = np.asarray(self.center)
        # pt2 = np.asarray(self.past[0].center)
        # x1, y1, x2, y2 = ballpt[0], ballpt[1], pt2[0], pt2[1]
        # coef = np.polyfit(past[:, 0], past[:, 1], deg=1)
        # a, b = coef[0], coef[1]
        # self.motion.coefficients = (a, b)
        # y1 = int(a*x1 + b)
        # y2 = int(a*x2 + b)
        # intercept = x2, y2

        uppery = int(table.playbox[0][1])
        lowery = int(table.playbox[1][1])

        collision_wall = False
        bouncelines = []

        # # if going left
        # if x1 < x2:
        #     arrowx = int(self.center[0] - vel)
        #     borderx = int(table.playbox[0][0] + self.radius)
        #     if arrowx > borderx:
        #         x = arrowx
        #         collision_wall = False
        #     else:
        #         x = int(table.playbox[0][0] + self.radius)
        #         collision_wall = 'left'
        #     y2 = int(x * a + b)
        #     arrowy = int(arrowx * a + b)
        #     tot_magnitude = norm(np.array((arrowx, arrowy)) - np.array(self.center))
        #     intercept = (x, y2)
        #
        # # if going right
        # elif x1 > x2:
        #     arrowx = int(self.center[0] + vel)
        #     borderx = int(table.playbox[1][0] - self.radius)
        #     if arrowx < borderx:
        #         x = arrowx
        #         collision_wall = False
        #     else:
        #         x = borderx
        #         collision_wall = 'right'
        #     y2 = int(x * a + b)
        #     arrowy = int(arrowx * a + b)
        #     tot_magnitude = norm(np.array((arrowx, arrowy)) - np.array(self.center))
        #     intercept = (x, y2)
        #
        # # if crossing upperbound before intersection with sides
        # if y2 < uppery:
        #     y2 = uppery + int(self.radius)
        #     x = int((y2 - b) / a)
        #     collision_wall = 'top'
        #     intercept = (x, y2)
        #
        # # if crossing lowerbound before intersection with sides
        # elif y2 > lowery:
        #     y2 = lowery - int(self.radius)
        #     x = int((y2 - b) / a)
        #     collision_wall = 'bottom'
        #     intercept = (x, y2)



        # collision_wall = self.motion.collision
        #
        # tot_magnitude = self.motion.magnitude

        intercept = self.check_initial_collision(past)
        self.motionlines.append([self.center, intercept])
        velocity = intercept - ballpt
        if self.motion.collision:
            bounceline = self.find_bounce(self.center, intercept)
            intercept2 = self.check_initial_collision(bounceline)
            newline = (intercept, intercept2)
            self.motionlines.append(newline)
            # if self.motion.collision:
            #     bounceline2 = self.find_bounce(intercept, intercept2)
            #     intercept3 = self.check_wall_collision(bounceline2)
            #     newline = (intercept2, intercept3)
            #     self.motionlines.append(newline)

        for i in range(len(self.motionlines)):
            line = self.motionlines[i]
            if i < len(self.motionlines)-1:
                # pass
                cv.line(cur_frame, tuple(line[0]), tuple(line[1]), (255, 255, 255), 2)
            else:
                # pass
                cv.arrowedLine(cur_frame, tuple(line[0]), tuple(line[1]), (255, 255, 255), 2, tipLength=0.05)

        self.checkballsinpath()

        # if collision_wall:
        #     col_magnitude = norm(np.array(intercept) - np.array(self.center))
        #     bounce_magnitude = tot_magnitude - col_magnitude
        #     wall = table.walls[collision_wall]
        #     v_corner = np.array(wall[0]) - intercept
        #     v_ball = ballpt - intercept
        #     orth1 = np.array((v_corner[1], -v_corner[0]))
        #     orth2 = np.array((-v_corner[1], v_corner[0]))
        #     dot_1 = orth1.dot(v_ball)
        #     dot_2 = orth2.dot(v_ball)
        #     v_mirror = [orth1, orth2][np.argmax(np.array((dot_1, dot_2)))]
        #     v_mirror_unitv = v_mirror / norm(v_mirror)
        #
        #     reflection = (2 * (v_ball.dot(v_mirror_unitv)) * v_mirror_unitv - v_ball)
        #
        #     reflection = reflection / norm(reflection) * bounce_magnitude
        #     end_point = (intercept + reflection).astype('int')
        #     cv.line(cur_frame, tuple(ballpt), tuple(intercept), (255, 255, 255), 2)
        #     # bounceline = [tuple(intercept), tuple(end_point)]
        #     # print('points', intercept, end_point)
        #     cv.arrowedLine(cur_frame, tuple(intercept), tuple(end_point), (255, 255, 255), 2, tipLength=0.05)
        #
        # else:
        #     cv.arrowedLine(cur_frame, tuple(ballpt), tuple(intercept), (255, 255, 255), 2, tipLength=0.05)
        if norm(velocity) == 0:
            velocity_unit = (0, 0)
        else:
            velocity_unit = velocity / norm(velocity)
        self.velocitypast.append(velocity_unit)
        # self.velocitypast = np.append(self.velocitypast, velocity_unit)

    # def ischaotic_old(self):
    #     if self.ismoving:
    #         pastlocs = [i.center for i in self.past]
    #         totdistlist = []
    #         xdistlist = []
    #         ydistlist = []
    #         for i in range(len(pastlocs) - 1):
    #             pt1 = pastlocs[i]
    #             pt2 = pastlocs[i+1]
    #             dists = distance(pt1, pt2, seperate=True, absolute=False)
    #             totdistlist.append(dists[0])
    #             xdistlist.append(dists[1])
    #             ydistlist.append(dists[2])
    #
    #         switchcount = 0
    #         if len(totdistlist) > 2:
    #             for j in range(len(totdistlist) - 1):
    #                 xdist1, xdist2 = xdistlist[j], xdistlist[j + 1]
    #                 ydist1, ydist2 = ydistlist[j], ydistlist[j + 1]
    #                 if abs(xdist1) > 15 and abs(xdist2) > 15:
    #                     if xdist1 / xdist2 < 0:
    #                         switchcount += 1
    #                 if abs(ydist1) > 15 and abs(ydist2) > 15:
    #                     if ydist1 / ydist2 < 0:
    #                         switchcount += 1
    #         # print('switches', switchcount)
    #         if switchcount > 4:
    #             table.watchlist.append(self.past[-1])
    #             self.variance = 250
    #
    #     #         xvar = statistics.variance(xdistlist)
    #     #         yvar = statistics.variance(ydistlist)
    #     #         self.variancehistory.append((xvar + yvar) / 2)
    #     # else:
    #     #     self.variancehistory.append(0)
    #     # self.variance = sum(self.variancehistory) / len(self.variancehistory)

    def ischaotic(self):
        if len(self.velocitypast) > 4:
            var = norm(self.velocitypast.arr.var(axis=0))
            self.variance = var
            # print('var', var)
            self.variancehistory = np.append(self.variancehistory, var)

    def checkballsinpath(self, lines=None):
        count = 0
        if lines is None:
            lines = self.motionlines
        if len(lines) != 0:
            for line in lines:
                count += 1
                print('count:', count)
                p1 = np.asarray(line[0])
                p2 = np.asarray(line[1])
                for ball in table.balls:
                    if not ball.center == self.center:
                        p3 = np.asarray(ball.center)
                        # TODO: check
                        if norm(p2 - p1) == 0:
                            dist = 1000000
                        else:
                            dist = abs(norm(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1))
                        if dist < self.radius + ball.radius:
                            ball.collisioncourse = True
                        else:
                            ball.collisioncourse = False

    def check_initial_collision(self, pts):
        pts = np.array(pts)
        x1, x2 = pts[-1, 0], pts[1, 0]
        # print('pts', pts)
        coef = np.polyfit(pts[:, 0], pts[:, 1], deg=1)
        a, b = coef[0], coef[1]
        y2 = int(a * x2 + b)
        # print('y2', y2)
        intercept = pts[-1]
        # print('defintercept', intercept)
        vel = self.motion.speed
        # if going left
        if x1 < x2:
            arrowx = int(self.center[0] - vel)
            borderx = int(table.playbox[0][0] + self.radius)
            if arrowx > borderx:
                x = arrowx
                self.motion.collision = False
            else:
                x = int(table.playbox[0][0] + self.radius)
                self.motion.collision = 'left'
            y2 = int(x * a + b)
            arrowy = int(arrowx * a + b)
            tot_magnitude = norm(np.array((arrowx, arrowy)) - np.array(self.center))
            intercept = (x, y2)

        # if going right
        elif x1 > x2:
            arrowx = int(self.center[0] + vel)
            borderx = int(table.playbox[1][0] - self.radius)
            if arrowx < borderx:
                x = arrowx
                self.motion.collision = False
            else:
                x = borderx
                self.motion.collision = 'right'
            y2 = int(x * a + b)
            arrowy = int(arrowx * a + b)
            tot_magnitude = norm(np.array((arrowx, arrowy)) - np.array(self.center))
            intercept = (x, y2)
        else:
            tot_magnitude = 0
            self.motion.collision = False

        self.motion.magnitude = tot_magnitude
        uppery = int(table.playbox[0][1])
        lowery = int(table.playbox[1][1])

        if y2 < uppery:
            y2 = uppery + int(self.radius)
            x = int((y2 - b) / a)
            self.motion.collision = 'top'
            intercept = (x, y2)

        # if crossing lowerbound before intersection with sides
        elif y2 > lowery:
            y2 = lowery - int(self.radius)
            x = int((y2 - b) / a)
            self.motion.collision = 'bottom'
            intercept = (x, y2)

        print('postintercept', intercept)
        print('collision', self.motion.collision)
        intercept = np.array(intercept)
        return intercept

    def check_wall_cross(self, pt1, pt2):
        x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
        if x1 < x2:
            arrowx = int(self.center[0] - vel)
            borderx = int(table.playbox[0][0] + self.radius)
            if arrowx > borderx:
                x = arrowx
                self.motion.collision = False
            else:
                x = int(table.playbox[0][0] + self.radius)
                self.motion.collision = 'left'
            y2 = int(x * a + b)
            arrowy = int(arrowx * a + b)
            tot_magnitude = norm(np.array((arrowx, arrowy)) - np.array(self.center))
            intercept = (x, y2)

        # if going right
        elif x1 > x2:
            arrowx = int(self.center[0] + vel)
            borderx = int(table.playbox[1][0] - self.radius)
            if arrowx < borderx:
                x = arrowx
                self.motion.collision = False
            else:
                x = borderx
                self.motion.collision = 'right'
            y2 = int(x * a + b)
            arrowy = int(arrowx * a + b)
            tot_magnitude = norm(np.array((arrowx, arrowy)) - np.array(self.center))
            intercept = (x, y2)
        else:
            tot_magnitude = 0
            self.motion.collision = False

        self.motion.magnitude = tot_magnitude
        uppery = int(table.playbox[0][1])
        lowery = int(table.playbox[1][1])

        if y2 < uppery:
            y2 = uppery + int(self.radius)
            x = int((y2 - b) / a)
            self.motion.collision = 'top'
            intercept = (x, y2)

        # if crossing lowerbound before intersection with sides
        elif y2 > lowery:
            y2 = lowery - int(self.radius)
            x = int((y2 - b) / a)
            self.motion.collision = 'bottom'
            intercept = (x, y2)

    def find_bounce(self, startpt, intercept):
        col_magnitude = norm(np.array(intercept) - np.array(startpt))
        bounce_magnitude = self.motion.magnitude - col_magnitude
        wall = table.walls[self.motion.collision]
        v_corner = np.array(wall[0]) - intercept
        v_ball = np.array(startpt) - intercept
        orth1 = np.array((v_corner[1], -v_corner[0]))
        orth2 = np.array((-v_corner[1], v_corner[0]))
        dot_1 = orth1.dot(v_ball)
        dot_2 = orth2.dot(v_ball)
        v_mirror = [orth1, orth2][np.argmax(np.array((dot_1, dot_2)))]
        v_mirror_unitv = v_mirror / norm(v_mirror)

        reflection = (2 * (v_ball.dot(v_mirror_unitv)) * v_mirror_unitv - v_ball)

        reflection = reflection / norm(reflection) * bounce_magnitude
        end_point = (intercept + reflection).astype('int')
        bounceline = [tuple(intercept), tuple(end_point)]
        return bounceline


class Buffer(list):
    def __init__(self, iterable=[], fullhist=True, maxlength=ballframebuffer):
        if not fullhist and len(iterable) > maxlength:
            iterable = iterable[-maxlength:]
        super().__init__(iterable)
        self.fullhist = fullhist
        self.maxlength = maxlength

    @property
    def recent(self):
        if self.isfull:
            return self[-self.maxlength:]
        else:
            return self

    @property
    def arr(self):
        return np.array(self)

    @property
    def r_arr(self):
        return np.array(self.recent)

    @property
    def isfull(self):
        return len(self) > self.maxlength

    def append(self, item):
        super().append(item)
        if self.isfull and not self.fullhist:
            self.pop(0)


class Motion:
    def __init__(self, vector=None, speed=None, direction=None, coefficients=None):
        self.vector = vector
        self.speed = speed
        self.direction = direction
        self.coefficients = coefficients
        self.collision = False


def distance(pt1, pt2, seperate=False, absolute=True):
    x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
    if absolute:
        xdist = abs(x1 - x2)
        ydist = abs(y1 - y2)
    else:
        # pt1 should be the newest point, pt2 the older
        xdist = x1 - x2
        ydist = y1 - y2
    if xdist == 0:
        dist = ydist
    elif ydist == 0:
        dist = xdist
    else:
        dist = sqrt((xdist ** 2) + (ydist ** 2))
    if seperate is True:
        return dist, xdist, ydist
    else:
        return dist


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


def color_diff(col1, col2):
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
    edged = cv.Canny(image, lower, upper)
    return edged


def pop(array, i):
    if i < len(array) - 1:
        array = np.concatenate([array[:i], array[i + 1:]])
    else:
        array = array[:i]
    return array


def play_frame():
    global framenum
    global cur_frame, cleanframe
    ret, frame = cap.read()
    print('frame:', framenum)
    if ret:

        # time.sleep(0.0001 * slowdown)
        cur_frame = frame
        cleanframe = frame.copy()
        # cv.imwrite('./cleanframe.png', cleanframe)
        # table.drawboxes(frame)
        table.find_balls(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        viewer.set_image(frame)
        fps.update()
        framenum += 1
    else:
        viewer.destructor_function()
        exit(9)


def stop_loop():
    cap.release()
    cv.destroyAllWindows()
    fps.stop()
    pr.disable()
    # pr.sort_stats('tottime')
    for i in debuglist:
        print(i)
    pr.print_stats(sort='time')
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def main():
    global table, framenum, viewer, fps, cap, pr
    np.set_printoptions(linewidth=200)
    cap = cv.VideoCapture(filepath)
    viewer = pyglview.Viewer(window_width=2000, window_height=1000, fullscreen=True, opengl_direct=True)
    pr = cProfile.Profile()
    pr.enable()
    fps = FPS().start()
    viewer.set_destructor(stop_loop)
    if cap.isOpened():
        table = Table(setting_num=table_detect_setting)
        cap.release()
        cap = cv.VideoCapture(filepath)
    else:
        print("error opening video")
    viewer.set_loop(play_frame)
    viewer.start()


main()
