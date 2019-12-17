from imutils.video import FPS
import cv2 as cv
import numpy as np
from math import sqrt
import cProfile
import pyglview
cv2 = cv

debug = False       # Debug makes playback much slower, but saves images of ball groups for each frame in frames folder
livegroups = True  # Displays the group circles live on playback

filepath = './clips/2019_PoolChamp_Clip11.mp4'

# Used to calculate the color difference of some potential ball to the table
table_color_bgr = (210, 140, 10)   # Green: (60, 105, 0)    Blue: (210, 140, 10)
# This is the average radius in pixels of a ball in the images. There is a findballsize parameter in find_balls
# that can be used to identify average size
ballsize = 15
# This is the number of past frames to keep the found circles for identifying trajectories and deciding which
# balls are real
ballframebuffer = 5
# This setting is for the canny and lines used to identify the table boundaries
table_detect_setting = 2  # 2 was working really well

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
        self.intersections = []
        self.setting = setting_num

        # self.circlehistory = []
        self.circles = []
        self.circlehistory = []
        self.grouped_circles = []
        # self.balls = []

        self.find_table_lines()
        self.find_all_intersections()
        self.goodpoints = self.check_all_intersections()
        self.set_boxes(self.goodpoints)
        check = self.drawlines(color=(0, 200, 255))
        if debug:
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

        if debug:
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

        if debug:
            cv.imwrite('./debug_images/4_horizontal_split.png', horiz)
            cv.imwrite('./debug_images/4_vertical_split.png', vert)

        self.linelist = {'top': toplines, 'bottom': bottomlines, 'left': leftlines, 'right': rightlines}
        print('Grouping lines by proximity and replacing them with an averaged line...')

        for key in self.linelist:
            lines = self.linelist[key]
            grouped = self.group_lines_by_proximity(lines)

            if key == 'top' or key == 'bottom':
                axis = 'y'
            else:
                axis = 'x'
            averaged = self.find_average_lines_from_groups(grouped, axis)
            self.linelist[key] = averaged

        if debug:
            checkimg = self.drawlines(frame.copy(), linelist=grouped, color=(0, 0, 255))
            cv.imwrite('./debug_images/5_' + key + 'lines.png', checkimg)
            checkimg = self.drawlines(frame.copy(), linelist=averaged, color=(0, 0, 255))
            cv.imwrite('./debug_images/6_averaged_' + key + 'lines.png', checkimg)
            copy2 = self.drawlines(frame.copy(), color=(0, 0, 255))
            cv.imwrite('./debug_images/7_grouped.png', copy2)

        print('Lines Found')
        return frame

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

        print('Checking line distances by part of table...')

        def get_relative_distances():
            distances = {'full': [], 'wood': [], 'bump': []}
            for key_ in goodkeys:
                if key_ == 'top':
                    ind = 1
                    top = linedict[key_]
                    playfield, pockets, table = self.max_mid_min(top, axis='y')
                elif key_ == 'bottom':
                    ind = 1
                    bottom = linedict[key_]
                    table, pockets, playfield = self.max_mid_min(bottom, axis='y')
                elif key_ == 'left':
                    ind = 0
                    left = linedict[key_]
                    playfield, pockets, table = self.max_mid_min(left, axis='x')
                elif key_ == 'right':
                    ind = 0
                    right = linedict[key_]
                    table, pockets, playfield = self.max_mid_min(right, axis='x')
                else:
                    print('line_dict is not in the right format in check_line_ratios')
                loc_play = playfield[ind]
                loc_poc = pockets[ind]
                loc_tab = table[ind]

                dist_tab_play = abs(loc_tab - loc_play)
                dist_tab_poc = abs(loc_tab - loc_poc)
                dist_poc_play = abs(loc_poc - loc_play)

                distances['full'].append(dist_tab_play)
                distances['wood'].append(dist_tab_poc)
                distances['bump'].append(dist_poc_play)
            avg_widths = {'full': [], 'wood': [], 'bump': []}
            for key_ in distances:
                dists = distances[key_]
                dev = np.std(dists)
                if dev > 7:  # TODO: may need to tune deviation
                    print('\ndeviaition is high in ' + key_)
                    print('you might wanna check the debug images.\n')
                else:
                    avg = sum(dists) / len(dists)
                    avg_widths[key_].append(int(avg))
            return avg_widths

        def check_against_avgs(checklines, xory, avg_dict, thresh=10):
            checklines = [list(i) for i in checklines]
            dist_list = [val[0] for val in avg_dict.values()]
            keeplines = []
            for i in range(len(checklines)):
                line = checklines[i]
                for j in range(i + 1, len(checklines)):                      # TODO: KEEP AND EYE OUT HERE FOR ISSUES
                    vs_line = checklines[j]
                    dist = abs(line[xory] - vs_line[xory])
                    for i in range(dist - thresh, dist + thresh):
                        if int(i) in dist_list:
                            if line not in keeplines:
                                keeplines.append(line)
                            if vs_line not in keeplines:
                                keeplines.append(vs_line)

        if linedict is None:
            linedict = self.linelist
        goodkeys = []
        badkeys = []
        for key in linedict:
            group = linedict[key]
            if len(group) == 3:
                goodkeys.append(key)
            else:
                badkeys.append(key)

        for key in badkeys:
            group = linedict[key]
            if len(group) < 3:
                print('Not enough lines found for the ' + key + ' side. You may need to change the setting or tune some'
                      ' parameters in Table.find_table_lines()')
            if len(group) > 3:
                if key == 'top' or key == 'bottom':
                    axis = 1
                elif key == 'left' or key == 'right':
                    axis = 0
                average_dists = get_relative_distances()
                check_against_avgs(group, axis, average_dists)

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

    def group_lines_by_proximity(self, linelist=None, thresh=30, group_num_thresh=3):
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
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles:
                x, y = circle[0], circle[1]
                radius = circle[2]
                cv.circle(frame, (x, y), radius, (0, 255, 0), 1)
                cv.circle(frame, (x, y), 2, (0, 0, 255), 2)
                rect = self.calculate_radius_square(radius, (x, y))
                cv.rectangle(frame, rect[0], rect[1], (255, 0, 255), 1)
        else:
            print('no circles to draw')
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
        if debug:
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
            pnts = [i[0] for i in p_w_d]
            good_groups = []
            quadrants = {'ul': [], 'ur': [], 'br': [], 'bl': []}
            for ptdist in p_w_d:
                point = ptdist[0]
                key = self.check_quadrant(point)
                quadrants[key].append(ptdist)
            # keylist = []
            for key in quadrants:
                quadrants[key].sort(key=lambda x: x[1])
                quadrants[key] = [i[0] for i in quadrants[key]]
                # keylist.append(key)
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

                if debug:
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
        if frame is None:
            frame = cur_frame
        copy = frame.copy()
        cropbox = self.pocketbox
        top, bottom = cropbox[0][1], cropbox[1][1]
        left, right = cropbox[0][0], cropbox[1][0]

        copy[: top], copy[bottom:] = (0, 0, 0), (0, 0, 0)
        copy[:, : left], copy[:, right:] = (0, 0, 0), (0, 0, 0)
        if debug:
            cv.imwrite('./debug_images/10_ball_area_crop.png', copy)

        gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 5)
        # thresh = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 15, 2)
        # cv.imwrite('./debug_images/10_adaptive_thresh.png', thresh)

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
            param1 = 80
            param2 = 30
            dp = 1.8

        if findballsize:
            minradius = 0
            max_radius = 50
            draw = True

        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=minradius, maxRadius=max_radius)

        if circles is not None:
            circles = circles[0]

            if draw:
                frame = self.draw_circles(circles, frame)
            if findballsize:
                self.get_ball_size(circles)

            self.add_circles_to_log(circles)
            self.average_motionblur_circles()
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
        groups = self.grouped_circles
        distthresh = 100
        regcolorthresh = 70
        blurcolorthresh = 110
        tablecolorthresh = 80
        # if len(history) > 0:
        circles = self.circles

        # At the start of the program, start a new group
        if len(groups) == 0:
            if len(history) > 0:

                print('called')
                oldcircles = history[-1]
                for circ in circles:
                    circs_w_dists = self.dist_list(circ, oldcircles)
                    if circs_w_dists[0][1] < distthresh:
                        closest = circs_w_dists[0][0]
                        newball = Ball(circ)
                        newball.append(closest)
                        newball.append(circ)
                        self.grouped_circles.append(newball)

        # Once there's a place to start for the group comparisons:
        else:
            oldcircles = [i[-1] for i in groups]
            ungrouped = []
            ballsappended = []
            for circ in circles:
                if circ.isblurred:
                    colorthresh = blurcolorthresh
                else:
                    colorthresh = regcolorthresh
                grouped = False
                circs_w_dists = self.dist_list(circ, oldcircles)
                for j in range(len(groups)):
                    closest = circs_w_dists[j][0]
                    distfrom = circs_w_dists[j][1]
                    colordiff = circ.compare_color(closest)
                    tablediff = color_difference(circ.color, table_color_bgr)
                    if tablediff < tablecolorthresh:
                        print(str(framenum), 'too close to table color')
                        break
                    elif colordiff < colorthresh:
                        for i in range(len(groups)):

                            if distfrom < distthresh * (groups[i].nmissingframes + 1):  # or distfrom < distthresh2:
                                if closest in groups[i]:
                                    groups[i].append(circ)
                                    grouped = True
                                    if groups[i].is_past_boundary(self.playbox):
                                        groups[i].inpocket = True
                                    break
                        if grouped is True:
                            break
                if not grouped:
                    print('frame:', framenum)
                    print(circ, 'not grouped. was looking for', closest)
                    print('distfrom:', distfrom, 'colordiff:', colordiff)
                    ungrouped.append(circ)
            self.grouped_circles = groups

            # colors = [(255, 0, 0), (255, 200, 0), (200, 255, 0), (0, 255, 0), (0, 255, 200), (0, 200, 255), (0, 0, 255), (200, 0, 255), (255, 0, 200), (0, 0, 0), (255, 255, 255), (100, 255, 0), (255, 100, 0), (0, 255, 100), (0, 100, 255)]
            colors = [ball.color for ball in groups]
            for ball in groups:
                ball.update_lastseen()

            if debug or livegroups:
                if debug and not livegroups:
                    copy = cur_frame.copy()
                else:
                    copy = cur_frame
                for i in range(len(groups)):
                    group = groups[i]
                    if group.inpocket:
                        color = (0, 255, 0)
                    else:
                        color = colors[i]
                    for c in group:
                        cv.circle(copy, c.center, c.radius, color, 3)
                for c in ungrouped:
                    cv.circle(copy, c.center, c.radius, (0, 0, 0), 2)
                cv.putText(copy, 'frame: ' + str(framenum), (20, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
                if debug:
                    cv.imwrite('./debug_images/frames/16_ungrouped_circles_' + str(framenum) + '.png', copy)

    def average_motionblur_circles(self, tolerance=2, colorthresh=40):
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
                    dist = get_dist(circle.center, circle2.center)
                    joined_radius = circle.radius + circle2.radius
                    if dist < joined_radius - tolerance and colordiff < colorthresh:
                        print(framenum, 'double circle found!!')
                        print('dist:', dist, 'joined radius:', joined_radius, 'color difference:', colordiff)
                        avgrad = joined_radius / 2
                        avgpoint = self.halfway_point(circle.center, circle2.center)
                        newcirc = Circle(avgpoint, int(avgrad))
                        self.circles[i] = newcirc
                        killlist.append(j)
        self.remove_list_of_indexes(self.circles, killlist)

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
    def dist_list(circ, oldcircles):
        circs_w_dists = []
        for ocirc in oldcircles:
            center1 = circ.center
            dist = get_dist(center1, ocirc.center)
            circs_w_dists.append((ocirc, dist))
        circs_w_dists.sort(key=lambda x: x[1])
        return circs_w_dists

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


class Circle:
    def __init__(self, center, radius):
        self.movementthresh = 10
        self.center = center
        self.radius = radius
        self.isblurred = False
        self.color = self.get_ball_color()

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

    def get_ball_color(self):
        tl, br = self.calculate_radius_square()
        x1, y1 = tl[0], tl[1]
        x2, y2 = br[0], br[1]
        cropped = cleanframe[y1: y2, x1: x2]
        if debug:
            cv.imwrite('./debug_images/ball_crop/15_cropped_ball.png', cropped)
        avg_b = int(np.mean(cropped[:, :, 0]))
        avg_g = int(np.mean(cropped[:, :, 1]))
        avg_r = int(np.mean(cropped[:, :, 2]))
        avg = (avg_b, avg_g, avg_r)
        # print(avg)
        return avg

    def compare_color(self, circ2):
        col1 = self.color
        col2 = circ2.color
        diff = color_difference(col1, col2)
        return diff


class Ball(Circle):
    def __init__(self, circle):
        self.past = []
        self.inpocket = None
        self.lastseenframe = None
        self.nmissingframes = 0
        super().__init__(circle.center, circle.radius)

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
        if len(self.past) > ballframebuffer:
            del self.past[0]

    def update_lastseen(self):
        self.nmissingframes = framenum - self.lastseenframe

    def is_past_boundary(self, box):
        x, y = self.center[0], self.center[1]
        xmin, ymin = box[0][0], box[0][1]
        xmax, ymax = box[1][0], box[1][1]
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return True
        else:
            return False


def stop_loop():
    cap.release()
    cv.destroyAllWindows()
    fps.stop()
    pr.disable()
    pr.print_stats()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def get_dist(pt1, pt2, seperate=False, absolute=True):
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
    edged = cv.Canny(image, lower, upper)
    return edged


def play_frame():
    global framenum
    global cur_frame, cleanframe
    ret, frame = cap.read()
    print('frame:', framenum)
    if ret:
        cur_frame = frame
        cleanframe = frame.copy()
        table.drawboxes(frame)
        table.find_balls(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        viewer.set_image(frame)
        fps.update()
        framenum += 1
    else:
        viewer.destructor_function()
        exit(9)


def main():
    global table, framenum, viewer, fps, cap, pr
    cap = cv.VideoCapture(filepath)
    viewer = pyglview.Viewer(window_width=2000, window_height=1000, fullscreen=False, opengl_direct=True)
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

