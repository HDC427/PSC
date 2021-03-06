import Predict
import cv2
import numpy as np
from util import match
from util import weighted_dist


def to_tuple_coord(coord):
    coord = np.array(coord)
    coord = np.array(coord+0.5, dtype=int)
    return tuple(coord)[::-1]


Skeleton_pair_list = [[5,7], [7,9], [5,11], [6,12], [6,8], [8,10], [11,12], [11,13], [13,15], [12,14], [14,16]]

def draw_skeleton(img, coord_list, conf_list, min_conf, color_p=(0,0,255), color_l=(255,0,0)):
    coord_list = [to_tuple_coord(coord) for coord in coord_list]
    
    validity_list = [False]*len(coord_list)
    validity_list[0] = True
    
    img = cv2.circle(img, coord_list[0], 3, color_p, 3)
    
    ## Draw the points and verify their validities.
    for i in range(5, len(coord_list)):
        if conf_list[i] >= min_conf:
            validity_list[i] = True
            img = cv2.circle(img, coord_list[i], 2, color_p, 2)
    
    ## Draw the shoulder and the head-line.
    if validity_list[5] and validity_list[6]:
        img = cv2.line(img, coord_list[5], coord_list[6], color_l, 2)
        coord_tmp = (np.array(coord_list[5]) + np.array(coord_list[6])) / 2
        coord_tmp = np.array(coord_tmp, dtype=int)
        coord_tmp = tuple(coord_tmp)
        img = cv2.line(img, coord_list[0], coord_tmp, color_l, 2)

    ## Draw all other pairs.
    for pair in Skeleton_pair_list:
        if validity_list[pair[0]] and validity_list[pair[1]]:
            img = cv2.line(img, coord_list[pair[0]], coord_list[pair[1]], color_l, 2)


    return img




class Skeleton:
    color_point = (0,0,255)
    color_skeleton = (255,0,0)

    ## List of pairs that we need to draw the line.
    ## [5,6] is not listed.
    Skeleton_pair_list = [[5,7], [7,9], [5,11], [6,12], [6,8], [8,10], [11,12], [11,13], [13,15], [12,14], [14,16]]

    threshold_dist = 35
    corr_vector = np.array([ 6.25851288e-01,  6.23950535e-01, -1.10839591e-03, -1.37307079e-04])
    

    def __init__(self, max_sample_size=10, sample_size=7, shape=(2,), order=3, min_score = 0.2, min_conf = 0.3):
        self.head_predictor = Predict.Predict(max_sample_size, sample_size, shape, order)
        #self.throat_predictor = Predict.Predict(max_sample_size, sample_size, shape, order)
        self.num_pass = 0

        self.coord = None
        self.score = None
        self.last_coord = None
        self.validity = False

        self.min_score = min_score
        self.min_conf = min_conf


    ###############################
    ## Function : match
    ##      Used to find the corressponding coord
    ##      of this skeleton.
    def match(self, tar_coord_list):
        obj_coord = self.head_predictor.predict()
        ind_l, conf_l = match(obj_coord, tar_coord_list)

        return ind_l, conf_l

    ##############################
    ## Function : corr_score
    ##      Used to calculate the confiance that this 
    ##      skeleton correspond to the new coord.
    def corr_score(self, coord, score):
        ## Calculate the 
        dist_w = self.weighted_distance(coord, score)
        dist_p = np.linalg.norm(self.head_predictor.predict()-coord[0])
        conf = conf(dist_p)

        return np.dot(self.corr_vector, np.array([1, conf, dist_p, dist_w]))


    



    def update(self, new_coord=None, new_score=None):
        ## if new_coord is valide.
        if not new_coord is None:
            self.coord = new_coord
            self.score = new_score
            self.last_coord = new_coord
            self.head_predictor.update(self.coord[0])
            self.validity = True
            self.num_pass = 0
        else:
            #self.coord = None
            #self.score = None
            self.head_predictor.push(None)
            self.num_pass += 1

        
    def invalid(self):
        self.validity = False
        self.head_predictor.re_init()
            

    def draw(self, img):
        ## If not valide,
        if not self.validity:
            return img
        
        ## or this skeleton is passed, we do not draw it.
        if self.num_pass > 0:
            return img

        img = draw_skeleton(img, self.coord, self.score, self.min_score, \
            self.color_point, self.color_skeleton)

        return img


    def weighted_distance(self, coord_new, score_new):
        if self.coord is None:
            return np.inf
        else:
            return weighted_dist(self.coord, self.score, coord_new, score_new)
