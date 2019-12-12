import cv2 as cv
import numpy as np
import imutils
import cython
import time
from math import sqrt

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
        self.intersections = []
        self.setting = setting_num

        self.find_table_lines()
        self.group_lines_by_category()
        self.find_all_intersections()
        self.check_all_intersections()
        self.set_boxes()
        check = self.drawlines(color=(0, 200, 255))
        cv.imwrite('./debug_images/10_final_lines.png', check)
        print('Table Initialized!')

    def find_table_lines(self):
        global shape
        if cap.isOpened():
            ret, frame = cap.read()
            self.frame = frame
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
        cv.imwrite('./debug_images/4_horizontal_split.png', horiz)
        vert = self.drawlines(frame.copy(), linelist=vertical, color=(0, 0, 255))
        cv.imwrite('./debug_images/4_vertical_split.png', vert)

        self.linelist = {'top': toplines, 'bottom': bottomlines, 'left': leftlines, 'right': rightlines}
        print('Grouping lines by proximity and replacing them with an averaged line...')

        for key in self.linelist:
            lines = self.linelist[key]
            grouped = self.group_lines_by_proximity(lines)
            checkimg = self.drawlines(frame.copy(), linelist=grouped, color=(0, 0, 255))
            cv.imwrite('./debug_images/5_' + key + 'lines.png', checkimg)
            if key == 'top' or key == 'bottom':
                axis = 'y'
            else:
                axis = 'x'
            averaged = self.find_average_lines_from_groups(grouped, axis)
            checkimg = self.drawlines(frame.copy(), linelist=averaged, color=(0, 0, 255))
            cv.imwrite('./debug_images/6_averaged_' + key + 'lines.png', checkimg)
            self.linelist[key] = averaged

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
        #
        # toplines = self.find_average_lines_from_groups(toplines, axis='y')
        # bottomlines = self.find_average_lines_from_groups(bottomlines, axis='y')
        # leftlines = self.find_average_lines_from_groups(leftlines, axis='x')
        # rightlines = self.find_average_lines_from_groups(rightlines, axis='x')
        #
        # top = self.drawlines(frame.copy(), linelist=toplines, color=(0, 0, 255))
        # cv.imwrite('./debug_images/6_toplines_avg.png', top)
        # bottom = self.drawlines(frame.copy(), linelist=bottomlines, color=(0, 0, 255))
        # cv.imwrite('./debug_images/6_bottomlines_avg.png', bottom)
        # left = self.drawlines(frame.copy(), linelist=leftlines, color=(0, 0, 255))
        # cv.imwrite('./debug_images/6_leftlines_avg.png', left)
        # right = self.drawlines(frame.copy(), linelist=rightlines, color=(0, 0, 255))
        # cv.imwrite('./debug_images/6_rightlines_avg.png', right)
        #
        # # self.linelist = [toplines, rightlines, bottomlines, leftlines]
        # self.linelist = {'top': toplines, 'bottom': bottomlines, 'left': leftlines, 'right': rightlines}

        # for key in self.linelist:
        #     print(key)

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
            # for lines in linelist:
            #     if isinstance(lines[0], (int, float, np.int32, np.int64)):
            #         x1, y1, x2, y2 = lines[0], lines[1], lines[2], lines[3]
            #         pt1, pt2 = (x1, y1), (x2, y2)
            #         cv.line(frame, pt1, pt2, color, thickness)
            #     else:
            #         for line in lines:
            #             if isinstance(line[0], (int, float, np.int32, np.int64)):
            #                 x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            #                 pt1, pt2 = (x1, y1), (x2, y2)
            #                 cv.line(frame, pt1, pt2, color, thickness)
        elif isinstance(linelist, dict):
            for key in linelist:
                lines = linelist[key]
                draw([lines])
        return frame

    def group_lines_by_category(self):

        lines = self.linelist
        lines = {key: np.asarray(value) for (key, value) in lines.items()}

        self.check_line_ratios(lines)       # TODO: CHECK LINE RATIOS


        # copy = self.frame.copy()
        # cv.circle(copy, test, 10, (0, 255, 255), 10)
        # cv.imwrite('./debug_images/99_coords_test.png', copy)

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

        playfield = self.drawlines(self.frame.copy(), playfield_lines, (0, 0, 255), 5)
        cv.imwrite('./debug_images/8_border_playfield.png', playfield)

        table_ = self.drawlines(self.frame.copy(), table_lines, (255, 0, 0), 5)
        cv.imwrite('./debug_images/8_border_table.png', table_)

        pockets = self.drawlines(self.frame.copy(), pocket_lines, (0, 255, 0), 5)
        cv.imwrite('./debug_images/8_border_pockets.png', pockets)

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
            print('\n')
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
            dist_list = [val [0] for val in avg_dict.values()]
            keeplines = []
            # keeplines.append(checklines[0])
            # print(len(checklines), 'lines to check distance on.')
            for i in range(len(checklines)):
                line = checklines[i]
                # print(type(line))
                # inlist = False
                for j in range(i + 1, len(checklines)):                      # TODO: KEEP AND EYE OUT HERE FOR ISSUES
                    vs_line = checklines[j]
                    dist = abs(line[xory] - vs_line[xory])
                    for i in range(dist - thresh, dist + thresh):
                        if int(i) in dist_list:
                            # print('YUP!')
                            if line not in keeplines:
                                keeplines.append(line)
                            if vs_line not in keeplines:
                                keeplines.append(vs_line)
                    # for val in dist_list:
                    #     diff = abs(dist - val)
                    #     print(diff)
                    #     if diff < thresh and line not in keeplines:
                    #         keeplines.append(line)
            # print('keeping', len(keeplines), 'lines.')
            # for i in keeplines:
            #     pass
                # print(i)

            # copy = self.drawlines(self.frame.copy(), keeplines, (255, 255, 255), 2)
            # cv.imwrite('./debug_images/11_line_dist_check.png', copy)

        # def check_ratios():


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
        goodlines = [linedict[key] for key in goodkeys]
        # check_ratios(goodlines)

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

    def find_contours_from_lines(self, linelist):
        print('\n')
        axis = self.check_axis(linelist[0])
        # print('len:', len(linelist))
        # print('axis:', axis)
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
                list_ = linelist[key]
        elif isinstance(linelist, (list, np.ndarray)):
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
        print('Set bounding-boxes for parts of table...')
        tablelines = self.tablelines
        pocketlines = self.pocketlines
        playlines = self.playfieldlines

        groups = {'table': tablelines, 'pocket': pocketlines, 'play': playlines}
        points = {'table': None, 'pocket': None, 'play': None}

        for key in groups:
            top_l = self.find_line_intersection(groups[key]['top'], groups[key]['left'])
            top_r = self.find_line_intersection(groups[key]['top'], groups[key]['right'])
            bot_l = self.find_line_intersection(groups[key]['bottom'], groups[key]['left'])
            bot_r = self.find_line_intersection(groups[key]['bottom'], groups[key]['right'])
            copy = self.frame.copy()
            cv.circle(copy, top_l, 5, (0, 0, 255), 4)
            cv.circle(copy, top_r, 5, (0, 0, 255), 4)
            cv.circle(copy, bot_l, 5, (0, 0, 255), 4)
            cv.circle(copy, bot_r, 5, (0, 0, 255), 4)
            cv.rectangle(copy, top_l, bot_r, (0, 255, 0), 2)
            cv.imwrite('./debug_images/9_'+ key + '_box_check.png', copy)
            points[key] = {'tl': top_l, 'br': bot_r, 'bl': bot_l, 'tr': top_r}

        self.tablebox = (points['table']['tl'], points['table']['br'])
        self.pocketbox = (points['pocket']['tl'], points['pocket']['br'])
        self.playbox = (points['play']['tl'], points['play']['br'])

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
        # print(self.intersections)
        # print(len(self.intersections), 'intersections found.')

    def check_all_intersections(self):

        midx = self.frame.shape[1] // 2
        midy = self.frame.shape[0] // 2

        # def get_center_offset(lst):
        #     array = np.asarray(lst)
        #     avgx = np.mean(array[: ,0])
        #     avgy = np.mean(array[:, 1])
        #     x_offset = int(avgx - midx)
        #     y_offset = int(avgy - midy)
        #     return x_offset, y_offset

        def get_dists_from_center(points):

            shifted = [(xy[0] - midx, xy[1] - midy) for xy in points]  # shifting all coordinates so their distance from center can be easily detirmined
            shifted.sort(key=lambda x: abs(x[0]))  # sorting the shifted list by the absolute value of x
            coords_w_dists = []
            all_x_dists = []
            all_y_dists = []
            # x_offset, y_offset = get_center_offset(points)
            # relative_center = (0 + x_offset, 0 + y_offset)
            # relative_center = (midx + x_offset, midy + y_offset)
            # copy = self.frame.copy()
            # cv.circle(copy, relative_center, 5, (255, 255, 255), 5)
            # cv.circle(copy, relative_center, 500, (255, 255, 255), 5)
            # cv.imwrite('./debug_images/13_relative_center.png', copy)
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
            # for i in coords_w_dists:
            #     print(i)
            xoffset = sum(all_x_dists) // len(all_x_dists)
            yoffset = sum(all_y_dists) // len(all_y_dists)

            #
            # for i in range(len(shifted)):
            #     coord = shifted[i]
            #     x, y = coord[0], coord[1]
            #     xdist = abs(0 - abs(x))
            #     ydist = abs(0 - abs(y))
            #     all_x_dists.append(xdist)
            #     all_y_dists.append(ydist)
            #     centerdist = sqrt((xdist ** 2) + (ydist ** 2))
            #     coord = (coord[0] + midx, coord[1] + midy)
            #     coord_dist = (coord, centerdist)
            #     coords_w_dists.append(coord_dist)
            # coords_w_dists.sort(key=lambda x: x[1])
            # for i in coords_w_dists:
            #     print(i)

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
            # print(out)
            return out

        def find_good_points(p_w_d):
            pnts = [i[0] for i in p_w_d]
            good_groups = []
            quadrants = {'ul': [], 'ur': [], 'br': [], 'bl': []}
            for ptdist in p_w_d:
                point = ptdist[0]
                key = check_quadrant(point)
                quadrants[key].append(ptdist)
            # keylist = []
            for key in quadrants:
                quadrants[key].sort(key=lambda x: x[1])
                quadrants[key] = [i[0] for i in quadrants[key]]
                # keylist.append(key)
            grouplist = []
            for i in range(3):                                               # For each or the 3 parts
                print('round', i)
                group = []                                                      # Create a new group of points
                copy = self.frame.copy()
                for key in quadrants:                                           # For each group in quadrants
                    print(quadrants)
                    pt = quadrants[key][0]

                    group.append(pt)                                                # add the smallest of the quadrant to group
                    cv.circle(copy, pt, 5, (255, 255, 255), 5)
                    del quadrants[key][0]                                           # delete the value that you just added from the quadrant

                grouplist.append(group)                                         # add the group you just made to the list of groups

                for key in quadrants:                                           # For each group in quadrants:
                    quadrant = quadrants[key]
                    killlist = []
                    for ind in range(len(quadrant)):                                # For each index in the current quadrant
                        point = quadrant[ind]
                        for group in grouplist:                                         # For each group in the list of already-added groups
                            for gpoint in group:                                            # For each known good point in that group
                                if ind not in killlist and isinline(gpoint, point):                                     # if the point in the current quadrant at the current index is inline with the good point
                                    killlist.append(ind)                                            # Add that index to the killlist
                    print('kill:', killlist)
                    quadrants[key] = self.remove_list_of_indexes(quadrants[key], killlist)   # remove all indexes in the killlist from the quadrant


                cv.imwrite('./debug_images/14_quadrant_minimum_point_check' + str(i) + '.png', copy)

            # for i in range(len(p_w_d)):
            #     if i + 4 <= len(p_w_d):                                    # TODO: Maybe sort points into quadrant lists
            #         quadrants = {'ul': [], 'ur': [], 'br': [], 'bl': []}   # TODO 1. Get 2 intersection points at a time and check their quadrants
            #         pt1, pt2 = pnts[i], pnts[i + 1]                                # TODO 2. Then get their distance from eachother
            #         dist1_2 = self.get_dist(pt1, pt2)                         # TODO 3. Then iterate through the rest of the points until you find a point in one of the other 2 quadrants
            #                                                                        # TODO 4. Check that the new point has a dist ratio with the initial points of close to 2:1
            #         pts = [pt1, pt2]
            #         for pt in pts:
            #             key = check_quadrant(pt)
            #             # if quadrants[key] is None:
            #                 # quadrants[key] = pt
            #             quadrants[key].append(pt)
            #             # else:
            #             #     print('quadrant not empty in check_ratios within check_all_intersections')
            #             #     print(quadrants)
            #         print(quadrants)
            #         for j in range(i + 1, len(pnts)):
            #             pt3 = pts[j]
            #             key = check_quadrant(pt3)
            #             if quadrants[key] is None:
            #                 dist1_3 = self.get_dist(pt3, pt1)
            #                 dist2_3 = self.get_dist(pt3, pt2)
            #                 dists = [dist1_2, dist1_3, dist2_3]
            #                 # short = min(dists)
            #                 # dists.remove(short)
            #                 # mid = min(dists)
            #                 # long = max(dists)
            #                 for ind in range(len(dists)):
            #                     ind2 = (ind + 1) % len(dists)
            #                     twodists = [dists[ind], dists[ind2]]
            #                     small = min(twodists)
            #                     big = max(twodists)
            #                     ratio = big / small
            #                     if abs(2 - ratio) < thresh:
            #                         print('ratio is good!!')
            #
            #

                    # group = [pnts[i], pnts[i + 1], pnts[i + 2], pnts[i + 3]]
                    # for pt in group:
                    #     point = pt[0]
                    #     dist = pt[1]
                    #     key = check_quadrant(point)
                    #     if quadrants[key] == '':
                    #         quadrants[key] = point
                    #     else:
                    #         print('point at ' + str(point) )
                    # print(group)

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
        find_good_points(points_w_dists)

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
