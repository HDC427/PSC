from functions import perform_histogram
from random import randint as rand
import Color
import cv2


class Unknown_Person:

    """ Class : Unknown_Person

            Used for the unknown persons and when the identification failes
    """
    def __init__(self):

        self.fst = "Unknown"
        self.snd = "Person"

        self.color = (0,255,0)


class Person:

    """ Class : Person

            Used to define a person, use to track one person in the video.
    """  
    delta = 25

    def __init__(self, fst, snd):

        self.fst = fst 
        self.snd = snd
        self.color = (rand(0,255), rand(0,255), rand(0,255))
            
        self.Color = Color.Color()
        self.kp = None
        self.prob = 0


    def put_histogram(self, image, coords):

        """ Function put_histogram

            Create the histogram of color of the person under it Color field
            * Params: image, coords
                - coords : the ROI where we want to create the histogram 
        """
        self.Color.update(image, coords)


    def load_model(self, model):
        
        """ Function load_model

            Loads the siamese model trained
            * Params: model
                - model : The siamese trained model
        """
    
        self.Color.load_model(model)


    def update(self, keypoint, confidence):

        """ Function update

            updates the persons's keypoints and confidence to their current value 
            * Params: keypoint, confidence
                - keypoint: the keypoints set given by get_correct_coords and processed by the Tracker
                - confidence : confidence value given by the model prediction and/or histogram corelation
        """

        self.kp, self.prob = keypoint, confidence

    def identifie_correl(self, image,  keypoints_list, keypoint_coords):

        """ Function identifie
            Identifie the person in the set of corrds given by PoseNet using histogram correlation
            * Params: image, keypoint_list
                - keypoint_list : should be be list of cross coords of the right shoulder and the 
                  left hip or the left shoulder and the right hip given by get_correct_coords
        """
    
        return self.Color.identifie_correl(image, keypoints_list, keypoint_coords)


    def identifie_model(self, image,  keypoints_list, keypoint_coords):
        
        """ Function identifie_model

            Identifie the person in the set of corrds given by PoseNet using the given model
            * Params: image, keypoint_list
                - keypoint_list : should be be list of cross coords of the right shoulder and the 
                  left hip or the left shoulder and the right hip given by get_correct_coords
        """

        return self.Color.identifie_model(image, keypoints_list, keypoint_coords)
        

    def identifie_cross(self, image, keypoints_list, keypoint_coords, lambda_factor = 0.6):

        """ Function identifie_cross

            Identifie the person in the set of corrds given by PoseNet using both methods
            * Params: image, keypoint_list, lambda_factor
                - keypoint_list : should be be list of cross coords of the right shoulder and the 
                  left hip or the left shoulder and the right hip given by get_correct_coords
                - lambda_factor: the mixing factor between histogram correlation and Siamese Network 
                  confdences
        """

        return self.Color.identifie_cross(image,  keypoints_list, keypoint_coords, lambda_factor = lambda_factor)

    
    def identifie_cross_2(self, image, keypoints_list, lambda_factor = 0.6):

        """ Function identifie_cross

            Identifie the person in the set of corrds given by PoseNet using both methods
            * Params: image, keypoint_list, lambda_factor
                - keypoint_list : should be be list of cross coords of the right shoulder and the 
                  left hip or the left shoulder and the right hip given by get_correct_coords
                - lambda_factor: the mixing factor between histogram correlation and Siamese Network 
                  confdences
        """

        return self.Color.identifie_cross_2(image, keypoints_list, lambda_factor = lambda_factor)
    
    def draw(self, img, thickness = 2, score_trace = False):

        """ Function draw

            Used to draw a square on the given image,
            with a random color assigned to Person
        """

        if(self.kp == None):
            print("No posenet keypoint were given or identification failed")
            return img

        y_min , y_max = int(min(self.kp[0][0],self.kp[1][0])) , int(max(self.kp[0][0],self.kp[1][0]))
        x_min , x_max = int(min(self.kp[0][1],self.kp[1][1])) , int(max(self.kp[0][1],self.kp[1][1]))

        d = Person.delta
        prob = self.prob
        color = self.color
        fst = self.fst
        snd = self.snd
        cv2.rectangle(img, (x_min-d,y_min-d), (x_max+d,y_max+d), color=color, thickness=thickness)

        if(score_trace):
            cv2.putText(img,'{} {} \n Prob = {}'.format(fst, snd, prob), 
                        (x_min,y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=1)

        return img



