import numpy as np
import cv2
import math
import time
##  Returns sine of an angle.
def sine(angle):
    return math.sin(math.radians(angle))

##  Returns cosine of an angle
def cosine(angle):
    return math.cos(math.radians(angle))

##  Reads an image from the specified filepath and converts it to Grayscale. Then applies binary thresholding
##  to the image.
def readImage(filePath):
    mazeImg = cv2.imread(filePath)
    grayImg = cv2.cvtColor(mazeImg, cv2.COLOR_BGR2GRAY)
    ret,binaryImage = cv2.threshold(grayImg,127,255,cv2.THRESH_BINARY)
    return binaryImage

def getPolar(radius, theta):
    x = radius * cosine(theta)
    y = radius * sine(theta)

    return int(x) + 1,int(y) + 1



def getCellValues(radius_low, radius_high, theta_low, theta_high):


    x1, y1 = getPolar(radius_low, theta_high)
    x2, y2 = getPolar(radius_low, theta_low)
    x3, y3 = getPolar(radius_high, theta_high)
    x4, y4 = getPolar(radius_high, theta_low)

    x_mid_theta_low, y_mid_theta_low = getPolar(radius_low,(theta_high - theta_low)/2 + theta_low )
    x_mid_theta_high, y_mid_theta_high = getPolar(radius_high, (theta_high - theta_low)/2 + theta_low)
    x_mid_rad_low, y_mid_rad_low = getPolar((radius_high - radius_low)/ 2 + radius_low, theta_low)
    x_mid_rad_high, y_mid_rad_high = getPolar((radius_high-radius_low)/2 + radius_low, theta_high)

    x_third_theta_high1, y_third_theta_high1 = getPolar(radius_high, (theta_high - theta_low)/3 + theta_low)
    x_third_theta_high2, y_third_theta_high2 = getPolar(radius_high, 2 * (theta_high - theta_low)/3 + theta_low)

    points = [(x_mid_theta_low, y_mid_theta_low), (x_third_theta_high1, y_third_theta_high1),(x_third_theta_high2, y_third_theta_high2), (x_mid_rad_low, y_mid_rad_low), (x_mid_rad_high, y_mid_rad_high)]


    start_x = int(min( x1, x2, x3, x4, x_mid_theta_low, x_mid_theta_high)) - 5
    finish_x = int(max(x1, x2, x3, x4, x_mid_theta_high, x_mid_theta_low)) + 5
    start_y = int(min(y1, y2 ,y3, y4, y_mid_theta_low, y_mid_theta_high)) - 5
    finish_y = int(max(y1, y2 ,y3, y4, y_mid_theta_high, y_mid_theta_low)) + 5

    return (start_x, finish_x), (start_y, finish_y), points

##  This function accepts the img, level and cell number of a particular cell and the size of the maze as input
##  arguments and returns the list of cells which are traversable from the specified cell.
def findNeighbours(img, level, cellnum, size):
    
    ############################# Add your Code Here ################################


    if size == 1:
        levels = 5
    else:
        levels = 7

    no_of_cells = {1: 6, 2: 12, 3: 24, 4: 24, 5: 24, 6: 48}
    directions = ['rad_down', 'rad_up1','rad_up2','theta_down', 'theta_up']

    if level == 6:
        dist_threshold = 1.5
    else:
        dist_threshold = 3

    if cellnum == no_of_cells[level]:
        neighbours_dict = {'theta_up': (level, 1), 'theta_down': (level, cellnum -1)}
    elif cellnum == 1:
        neighbours_dict = {'theta_up': (level, cellnum + 1), 'theta_down': (level, no_of_cells[level])}
    else:
        neighbours_dict = {'theta_up': (level, cellnum + 1), 'theta_down': (level, cellnum -1)}


    if level == 2 or level == 6:
        neighbours_dict['rad_down'] = (level -1, (cellnum + 1)/2)
        neighbours_dict['rad_up1'] = (level + 1, cellnum * 2 - 1)
        neighbours_dict['rad_up2'] = (level + 1, cellnum * 2)
    elif level == 1:
        neighbours_dict['rad_down'] = (0, 0)
        neighbours_dict['rad_up1'] = (level + 1, cellnum * 2 - 1)
        neighbours_dict['rad_up2'] = (level + 1, cellnum * 2)
    elif level == 3:
        neighbours_dict['rad_down'] = (level -1, (cellnum + 1)/2)
        neighbours_dict['rad_up1'] = (level + 1, cellnum )
        neighbours_dict['rad_up2'] = (level + 1, cellnum )
    elif level == 5:
        neighbours_dict['rad_down'] = (level -1 , cellnum)
        neighbours_dict['rad_up1'] = (level + 1, cellnum * 2 - 1)
        neighbours_dict['rad_up2'] = (level + 1, cellnum * 2)
    else:
        neighbours_dict['rad_down'] = (level -1 , cellnum)
        neighbours_dict['rad_up1'] = (level + 1, cellnum )
        neighbours_dict['rad_up2'] = (level + 1, cellnum )

    radius_low = (level * 40)
    radius_high = radius_low + 40
    angle = (float(360 )/ no_of_cells[level])
    theta_low = (angle ) * (cellnum - 1)
    theta_high = theta_low + ( angle)

    x, y, points = getCellValues(radius_low, radius_high, theta_low, theta_high)
    x, y = shift_co_ordinates(x, y, size)

    x_points = [point[0] for point in points]
    y_points = [point[1] for point in points]
    x_points, y_points = shift_co_ordinates(x_points, y_points, size)
    points = [(x_points[i], y_points[i] ) for i in range(len(x_points))]
    points = [(point[0] - x[0] , point[1] - y[0]) for point in points]
    direction_dict = { directions[i]:points[i] for i in range(len(points)) }

    cell_img = img[y[0] : y[1], x[0] : x[1] ]

    cell_img = 255 - cell_img

    contours, hierarchy = cv2.findContours(cell_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        for point in points:
            dist = abs(cv2.pointPolygonTest(contour, point, True))
            cv2.circle(cell_img, point, 3, 255, 2)
            if dist < 5:
                
                if point == direction_dict['theta_up']:
                    neighbours_dict.pop('theta_up')
                if point == direction_dict['theta_down']:
                    neighbours_dict.pop('theta_down')
                if point == direction_dict['rad_down']:
                    neighbours_dict.pop('rad_down')
                if point == direction_dict['rad_up1']:
                    neighbours_dict.pop('rad_up1')
                if point == direction_dict['rad_up2']:
                    neighbours_dict.pop('rad_up2')
      

    neighbours = list(set(neighbours_dict.values()))

    


    #################################################################################
    return neighbours

##  colourCell function takes 5 arguments:-
##            img - input image
##            level - level of cell to be coloured
##            cellnum - cell number of cell to be coloured
##            size - size of maze
##            colourVal - the intensity of the colour.
##  colourCell basically highlights the given cell by painting it with the given colourVal. Care should be taken that
##  the function doesn't paint over the black walls and only paints the empty spaces. This function returns the image
##  with the painted cell.
def colourCell(img, level, cellnum, size, colourVal):
    ############################# Add your Code Here ################################

    if level == 0:
        return img

    if size == 1:
        levels = 5
        length = 440
    else:
        levels = 7
        length = 600

    no_of_cells = {1: 6, 2: 12, 3: 24, 4: 24, 5: 24, 6: 48}

    radius_low = (level * 40)
    radius_high = radius_low + 40
    angle = (360.0 / no_of_cells[level])
    theta_low = (angle ) * (cellnum - 1)
    theta_high = theta_low + ( angle)

    x, y, points = getCellValues(radius_low, radius_high, theta_low, theta_high)


    for x_polar in range(x[0], x[1] + 1, 1):
        for y_polar in range(y[0], y[1] +1, 1):
            radius = ( x_polar**2 + y_polar**2)**0.5
            theta = math.degrees(math.acos((x_polar/ radius)))
            if y_polar < 0:
                theta = (360 - theta) 
            if radius < radius_low or radius > radius_high:
                continue
            if theta < theta_low or theta > theta_high:
                continue
            x_cart, y_cart= shift_co_ordinates((x_polar,), (y_polar,), size)
            if img[y_cart[0], x_cart[0]] > 127:
                img[y_cart[0], x_cart[0]] = colourVal


    #################################################################################
    return img

def shift_co_ordinates(x, y, size):
    if size == 1:
        x = tuple( (item + 440/2) for item in x)
        y = tuple( (item + 440/2) for item in y)
    else:
        x = tuple( (item + 600/2) for item in x)
        y = tuple( (item + 600/2) for item in y)

    return x, y

##  Function that accepts some arguments from user and returns the graph of the maze image.
def buildGraph(img, size  ):   
    graph = {} 

    if size == 1:
        levels = 5
    else:
        levels = 7

    no_of_cells = {1: 6, 2: 12, 3: 24, 4: 24, 5: 24, 6: 48}

    for level in range(1, levels):
        for cell in range(1,no_of_cells[level] + 1):
            neighbours = findNeighbours(img, level, cell, size)
            graph[(level, cell)] = neighbours
            
   


    #################################################################################
    return graph


##  Function accepts some arguments and returns the Start coordinates of the maze.
def findStartPoint(img , size ):
    if size == 1:
        length = 440
        levels = 5
    else:
        length = 600
        levels = 7

    no_of_cells = {1: 6, 2: 12, 3: 24, 4: 24, 5: 24, 6: 48}

    img_inv = 255 - img
    start_img = np.copy(img)
    cv2.circle(start_img, (length/2, length/2), levels * 40, 0 ,3)
    
    start_img = ~(start_img | img_inv)
    contours, hierarchy = cv2.findContours(start_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [ cv2.contourArea(contour) for contour in contours]

    break_point = contours[areas.index(max(areas))]
    M = cv2.moments(break_point)
    centroid_x = int(M['m10']/M['m00'])
    centroid_y = int(M['m01']/M['m00'])
    centroid_x, centroid_y = centroid_x - length/2, centroid_y - length/2
    theta = math.degrees(math.acos((centroid_x/ float(levels * 40))))
    if centroid_y < 0:
        theta = (360.0 - theta) 
    level = levels - 1
    cellnum = int(theta/(360.0 /(no_of_cells[levels -1]))) + 1
    start = (level, cellnum)

    #################################################################################
    return start

##  Finds shortest path between two coordinates in the maze. Returns a set of coordinates from initial point
##  to final point.
def findPath(graph, start, end, path =[]):      ## You can pass your own arguments in this space.
    ############################# Add your Code Here ################################
    path = path + [start]
    # print path

    if start == end:
        return path

    if not graph.has_key(start):
        return None
    shortest = None

    for node in graph[start]:
        if node not in path:
            newpath = findPath(graph, node, end, path)

            if newpath:
                if (not shortest) or len(newpath) < len(shortest):
                    shortest = newpath[:]


    #################################################################################
    return shortest

##  This is the main function where all other functions are called. It accepts filepath
##  of an image as input. You are not allowed to change any code in this function. You are
##  You are only allowed to change the parameters of the buildGraph, findStartPoint and findPath functions
def main(filePath, flag = 0):
    img = readImage(filePath)     ## Read image with specified filepath
    if len(img) == 440:           ## Dimensions of smaller maze image are 440x440
        size = 1
    else:
        size = 2
    maze_graph = buildGraph(img, size )   ## Build graph from maze image. Pass arguments as required
    start = findStartPoint(img, size)  ## Returns the coordinates of the start of the maze
    shortestPath = findPath(maze_graph, start, (0,0))  ## Find shortest path. Pass arguments as required.
    print shortestPath
    string = str(shortestPath) + "\n"
    for i in shortestPath:               ## Loop to paint the solution path.
        img = colourCell(img, i[0], i[1], size, 127)
    if __name__ == '__main__':     ## Return value for main() function.
        return img
    else:
        if flag == 0:
            return string
        else:
            return graph
## The main() function is called here. Specify the filepath of image in the space given.
if __name__ == "__main__":
    filepath = "test/image_big.jpg"     ## File path for test image
    img = main(filepath)          ## Main function call
 
    cv2.imshow("image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
