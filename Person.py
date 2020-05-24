###
## Class : Person
##      Used to define, store and track one person in the video.

import numpy as np
import util
import Predict
import confiance

from CorrScore import CorrScore as cs
from Color.Color import Color

class Person:
    ###############
    # Global variables :
    color_point = (0,0,255)     # Default color of points.
    color_line = (255,0,0)      # Default color of lines.

    name = 'NO_NAME'

    ## List of pairs that we need to draw the line.
    ## [5,6] is not listed.
    Skeleton_pair_list = [[5,7], [7,9], [5,11], [6,12], [6,8], [8,10], [11,12], [11,13], [13,15], [12,14], [14,16]]

    #threshold_dist = 35         # May not be useful.
    # The vector used to calculate the corresponding score (corr_score).
    corr_vector = np.array([ 6.25851288e-01,  6.23950535e-01, -1.10839591e-03, -1.37307079e-04])


    ###############
    ## Functions :


    def __init__(self, max_sample_size=10, sample_size=7, shape=(2,), deg=3, min_score=0.2, min_conf=0.3):
        '''
        Constructor,
        * Param(s)
           max_sample_size     int, default is 10. The max number of samples we will store.
                                                   used in the construct of Predictor.
           sample_size         int, default is 7. The default sample size to do the regression.
                                                   used in the construct of Predictor.
           shape               tuple, default is (2,). The shape of each coord unit.
                                                   used in the construct of Predictor.
           deg                 int, default is 3. The default degree used in regression.
                                                   used in the construct of Predictor.
           min_score           double, default is 0.2. The minimum score used in drawing.
           min_conf            double, default is 0.3.

        '''
        # Define the predictor.
        self.predictor = Predict.Predict(max_sample_size, sample_size, shape, deg)
        self.num_pass = 0       # The number of frames that this person has lost its position.

        self.char_color = Color()

        self.coord = None       # Current coord.
        self.score = None       # Current score.
        self.key_point = None   # Key points used in Color. # Not useful anymore, we compute it in get_key_point_coords and after, we loose it
        self.prob = 0           # used in Color
        #self.last_coord = None  # The coord at time t-1.
        self.validity = False  # If this person losses its position, its not valid.

        self.min_score = min_score  # minimum score, used in drawing.
        self.min_conf = min_conf    # minimum confiance, used in matching.


    def match(self, tar_coord_list):
        '''
        Function : match
            Given a list of targets, the function returns the sorted 
            target list, with confiances.
        * Param(s) : 
            tar_coord_list, numpy.ndarray, the list of targets.
        * Return(s) :
            ind_l, numpy.ndarray, the list of indices of targets, sorted 
                                    by confiance in descending order.
            conf_l, numpy.ndarray, the list of confiances. corresponding to
                                    the $ind_l.

        '''
        obj_coord = self.predictor.predict() # Get the object.
        ind_l, conf_l = util.match(obj_coord, tar_coord_list)

        return ind_l, conf_l



    ###
    def weighted_distance(self, coord_new, score_new):
        '''
        Function : weighted_distance
            Used to calculate the distance between the given coord and 
            the current coord. Weighted by the scores.
        * Param(s) : 
            coord_new, numpy.ndarray, the given new coord.
            score_new, numpy.ndarray, the given new score.
        * Return(s) :
            dist, double
        '''
        if self.coord is None:
            return np.inf
        else:
            return util.weighted_dist(self.coord, self.score, coord_new, score_new)

    ###
    def corr_score(self, coord, score):
        '''
        Function : corr_score
            Used to calculate the confiance that the given coord
            corresponds to this person.
        * Param(s) :
            coord, numpy.ndarray, the coord of target coord.
            score, numpy.ndarray, the score of target coord.
        * Return(s) :
            conf, double, the confiance. normally it is between 0 and 1.

        '''
        dist_w = self.weighted_distance(coord, score) # The weighted distance.
        dist_p = np.linalg.norm(self.predictor.predict() - coord[0]) # The distance of prediction.
        conf = confiance.conf(dist_p) # The confiance of prediction.
        
        #feature = np.array([1,conf, dist_p, dist_w])
        feature = np.array([conf, dist_p, dist_w])
        ret = cs.corr_score(feature)
        return ret[1]

    ###
    def update(self, new_coord=None, new_score=None):
        '''
        Function : update.
            Used to upgrade the new coord, score of this person.
            May be modified to update the parameters.
            If the new data are not None, this person will be 
            valid after this update.
        * Param(s) :
            new_coord, numpy.ndarray, default is None.
            new_score, numpy.ndarray, default is None.
        * Return(s) :

        '''
        ## if new_coord is valide.
        if not new_coord is None:
            self.coord = new_coord
            self.score = new_score
            self.predictor.update(self.coord[0])
            self.validity = True # Shall be valid.
            self.num_pass = 0    # Make the pass counter 0.
        else:
            #self.head_predictor.push(None)
            self.num_pass += 1
            # We make it invalid even just once miss.
            self.validity = False   

    def set_key_point(self, key_point, color_conf):
        self.key_point = key_point
        self.prob = color_conf


    def identifie_cross(self, image,  keypoints_list, keypoint_coords, lambda_factor=0.8):

        """ Function identifie_cross

            Identifie the person in the set of corrds given by PoseNet using both methods
            * Params: image, keypoint_list, lambda_factor
                - keypoint_list : should be be list of cross coords of the right shoulder and the 
                  left hip or the left shoulder and the right hip given by get_correct_coords
                - lambda_factor: the mixing factor between histogram correlation and Siamese Network 
                  confdences
        """
        return self.char_color.identifie_cross(image, keypoints_list, keypoint_coords, lambda_factor=lambda_factor)

    ###
    def draw(self, img):
        '''
        Function : draw
            Used to draw the skeleton in the given image,
            with specified point and line colors.

        '''
        # If this person is not valid or he is passed, we do not draw.
        if not self.validity:
            return img
        if self.num_pass > 0:
            return img
        
        img = util.draw_skeleton(img, self.coord, self.score, self.min_score, \
                self.color_point, self.color_line)

        return img
        

    def put_histogram(self, img, coords):

        """ Function put_histogram

            Create the histogram of color of the person under it Color field
            * Params: image, coords
                - coords : the ROI where we want to create the histogram 
        """
        self.char_color.update(img, coords)
