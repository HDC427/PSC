###
#
# @license
# This project is built based on the open-source projects: 
#   -> tfjs-models          https://github.com/tensorflow/tfjs-models
#   -> posenet-tensorflow   https://github.com/rwightman/posenet-pytorch
#
# This project is built by the psc-project group in Ecole Polytechnique.
# Cooperating with UbiSoft, this project may contains sensitive or business information.
# These information are protected by the NDA.
# All rights are reserved to the original team.
# 
# Last modified: 18:00, 19 Sep. 2019, by ZHONG Benxin
# 
#
###

'''
* These shall be re-coded using CUDA and Numba.
* The first version.
* 
0	nose
1	leftEye
2	rightEye
3	leftEar
4	rightEar
5	leftShoulder
6	rightShoulder
7	leftElbow
8	rightElbow
9	leftWrist
10	rightWrist
11	leftHip
12	rightHip
13	leftKnee
14	rightKnee
15	leftAnkle
16	rightAnkle
'''




import cv2
import numpy as np
import math
import colorsys 

COLOR_RED       = (255, 0, 0)
COLOR_BLUE      = (0, 0, 255)
COLOR_GREEN     = (0, 255, 0)
COLOR_YELLOW    = (255, 255, 0)
COLOR_VIOLET    = (255, 0, 255)


##
# Function: draw_skeleton
# Params: 
#           img,                the image in which we will draw
#           key_point_coords,   np.array of shape (17,2)
#           key_point_scores,   list/array of length 17
#           min_pose_score,
#           min_part_score,
#
# The function used to draw the skeleton, given the coordinates and the scores.
def draw_skeleton(img, key_point_coords, key_point_scores, min_pose_score=0.1, min_part_score=0.15, 
                colors = [COLOR_RED, COLOR_YELLOW, COLOR_VIOLET]):
    ##Transform the coordinates to np.array, with in type.
    key_point_coords = np.array(key_point_coords + (0.5*np.ones_like(key_point_coords)), dtype=int)
    key_point_scores = np.array(key_point_scores)
    if key_point_coords.shape != (17, 2):
        raise Exception('Error! The shape of key_point_coords should be (17, 2).')
    if not (key_point_scores.shape == (17,)):
        raise Exception('Error! The shape of key_point_scores should be (17,).')

    ##The nose and head:
    if key_point_scores[0] >= min_part_score:

        color = tuple(map(int, colors[0]))
        img = cv2.circle(img, tuple(key_point_coords[0])[::-1], 3, color=color, thickness=3)

        ### head
        # The radius of head cannot be well-determined jet.
        # We shall use the distances between eyes, nose and ears.
        '''
        distance_left = np.linalg.norm(key_point_coords[0]-key_point_coords[3])
        distance_right = np.linalg.norm(key_point_coords[0]-key_point_coords[4])
        r = distance_left if distance_left < distance_right else distance_right 
        img = cv2.circle(img, tuple(key_point_coords[0])[::-1], int(r+0.5), COLOR_RED, 3)
        '''

    for i in range(5, 17): 
        color = colors[1] if i%2 == 0 else colors[2]
        color = tuple(map(int, color))
        if key_point_scores[i] > min_part_score:
            img = cv2.circle(img, tuple(key_point_coords[i])[::-1], 3, color, 3)



    return img





class Person:
    _current_coord = np.zeros((17,2))
    _current_score = np.zeros(17)




#########################
# Class name: Trace
# 
# Used to predict the success coordinate of an object.
# Using the polynomial degression, numpy.polyfit, with 
# given size of sample and degree.
#
# Methodes:
#      update(self, new_coord): 
#                   Given the coordinate $n$, the function will 
#                   predict and return the coordinate $n+1$.
#                   -If we do not have yet received $sample_size$ many
#                   data, it will return $None$.
class Trace:
    sample_size = 10
    order = 2
    dimenssion = 2
    number = 1
    samples = np.zeros((0, 2))


    @staticmethod
    def dist(r0=np.zeros(2), r1=np.zeros(2)):
        r0 = np.array(r0)
        r1 = np.array(r1)

        if r0.shape != r1.shape:
            raise Exception('The shape of r0 does not match that of r1.')
        else:
            return max(abs(r0-r1))


    def __init__(self, sample_size=10, order=2, dimenssion=2, number=1):
        self.sample_size = sample_size
        self.order = order
        self.dimenssion = dimenssion
        self.number = number
        self.samples = np.zeros((0, number*dimenssion))

        # Construct the coefficients for predict.
        self.t = self.sample_size**(np.array(range(self.order+1), dtype=np.float64))
        self.t = self.t[::-1]


    def update(self, new_coord):
        new_coord = np.array(new_coord)
        # Reshape, as a 1-D vector. As required for numpy for polyfit.
        new_coord = new_coord.reshape(self.number*self.dimenssion)
        '''
        if new_coord.shape != (self.dimenssion,):
            raise Exception('The dimenssion of new_coord ' + str(new_coord.shape) \
                +  ' does not match the requirement (' + self.dimenssion + ',).')
        '''
        
        # Add the new coord.
        self.samples = np.vstack([self.samples, new_coord])

        ##
        # if we have not yet received $sample_size$ many samples,
        if len(self.samples) < self.sample_size:
            return len(self.samples)
        # we have already at least $sample_size$ many samples
        else:
            diff = len(self.samples) - self.sample_size 
            self.samples = self.samples[diff:]

            poly = np.polyfit(x=np.array(range(self.sample_size)),\
                            y=self.samples, deg=self.order)
            
            # The predicted value.
            value_p = np.dot(t, poly)
            # Reshape as $number$ many coords of $dimension$ dimension.
            value_p = value_p.reshape((self.number, self.dimenssion))

            return value_p






class SkltDrawer:
    _max_person_num = 4
    _person_list = []
    _min_part_score = 0.1
    _min_pose_score = 0.15
    _colors = []



    ####################################################
    ## DEBUG
    _trace_nose = Trace(order=2, sample_size=7)
    _coord_nose = np.zeros(2)
    _coord_nose_p = 0
    _ind_0 = 0
    ####################################################

    def __init__(self, max_person_num = 4, min_part_score = 0.1, min_pose_score = 0.15):
        self._max_person_num = max_person_num
        self._min_part_score = min_part_score
        self._min_pose_score = min_pose_score

        # Define the color scheme
        phases = np.array(range(self._max_person_num), dtype=np.uint8)*180/self._max_person_num
        colors = np.zeros((self._max_person_num, 3, 3), dtype=np.uint8)
        colors[:, 0, 0] = phases
        colors[:, 0, 1] = 255*np.ones_like(colors[:,0,1])
        colors[:, 0, 2] = 255*np.ones_like(colors[:,0,2])
        colors[:, 1, 0] = phases
        colors[:, 1, 1] = 255*np.ones_like(colors[:,0,2])
        colors[:, 1, 2] = 96*np.ones_like(colors[:,0,2])
        colors[:, 2, 0] = phases
        colors[:, 2, 1] = 96*np.ones_like(colors[:,0,2])
        colors[:, 2, 2] = 255*np.ones_like(colors[:,0,2])


        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2RGB)
        self._colors = np.array(colors, dtype=int)


    def stabelise_nose(self, current_coord, new_coord, threshold = math.pi):
        pass
        
    
    @staticmethod
    def to_tuple_coord(coord):
        coord = np.array(coord)
        coord = np.array(coord+0.5, dtype=int)
        return tuple(coord)[::-1]


    @staticmethod
    def dist(coords_1, coords_2, norm_style = 0):
        coords_1 = np.array(coords_1)
        coords_2 = np.array(coords_2)
        if coords_1.shape != coords_2.shape:
            raise Exception('The shapes of these two coords are different.' \
                            + 'Coord1 ' + str(coords_1.shape) \
                            + 'Coord2 ' + str(coords_2.shape))

        
        diff = coords_1 - coords_2
        if norm_style == 0:
            return np.linalg.norm(diff)

        norms = np.linalg.norm(diff, axis=1)
        if norm_style == 1:
            return max(norms)



    def get_id0_coords(self, key_point_coords, key_point_scores):
        # If we do not have the prediction.
        if not isinstance(self._coord_nose_p, np.ndarray):
            return self._ind_0
        
        # Pre-treat.
        key_point_coords = np.array(key_point_coords)
        coords_nose = key_point_coords[:,0]
        # Difference of vectors.
        coords_nose -= self._coord_nose_p
        # Calculate distances (Using norm normal).
        distances = np.linalg.norm(coords_nose, axis=1)
        # Find the minimum distance, and its index(es).
        self._ind_0 = np.argmin(distances)
        return self._ind_0




    def draw(self, img, key_point_coords, key_point_scores):
        if len(key_point_coords) != self._max_person_num:
            raise Exception('Wrong coordinate number of persons!') 
        if len(key_point_scores) != self._max_person_num:
            raise Exception('Wrong score number of persons!') 
        
        
        # Get the index of identity 0.
        self.get_id0_coords(key_point_coords, key_point_scores)
        # Update the coordinate info for id_0, and get the successeur prediction.
        self._coord_nose_p = self._trace_nose.update(key_point_coords[self._ind_0][0])
        

        base = np.array([100, 100])
        coord_base = (100, 100)

        direction = np.array(key_point_coords[self._ind_0][0]) - self._coord_nose
        direction *= 10
        direction += base
        
        direction = SkltDrawer.to_tuple_coord(direction)
        img = cv2.arrowedLine(img, coord_base, direction, color=COLOR_BLUE, thickness=3)
        self._coord_nose = key_point_coords[self._ind_0][0]

        ####
        # Test the class Trace
        # who predicts the coordinate using poly-fit.
        #coord_predict = self._trace_nose.update(key_point_coords[0][0])
        #coord_predict = SkltDrawer.to_tuple_coord(coord_predict)
        #img = cv2.circle(img, coord_predict, radius=3, color=COLOR_BLUE, thickness=3)
        
        img = draw_skeleton(img, key_point_coords[self._ind_0], key_point_scores[self._ind_0],\
                            colors=self._colors[0])
        '''
        for i in range(self._max_person_num):
            img = draw_skeleton(img, key_point_coords[i], key_point_scores[i],
                colors=self._colors[i])
        '''

        return img