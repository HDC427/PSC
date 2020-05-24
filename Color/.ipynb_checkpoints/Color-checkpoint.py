import numpy as np
import cv2
import matplotlib.pyplot as plt
from functions import perform_histogram


class Color:
    
    """ Class name: Color

        Used to create and store color signature of a Person instance
        Using the color density repartition and CNN model; cv2.calcHist, Siamese Network 
    """

    def __init__(self):

        self.histogram = None  # for histogram_correlation calcul
        self.histogram_reshaped = None  # for CNN model prediction
        self.model = None    # CNN model to be use (Siamese Network of input_shape = (30,30))

    
    def update(self, frame, coords):
        
        """ Function update

            Upates the histogram of Color of the person from a ROI
            * Params: frame, coords
                - coords : given by get_correct_coords
            * Returns: void
        """
        
        # create the histogram on a ROI of the input frame
        self.histogram = perform_histogram(frame,coords) 
        # reshape the histogram for the model prediction
        self.histogram_reshaped = self.histogram.reshape(1,30,30,1)


    def update_histogram(self, histogram, reshaped = False):

        """" Function update_histogram

             Upates the histogram of Color of the person from a given histogram
             * Params: frame, histogram
                - histogram : histogram to be attributed
             * Returns: void
        """
        # set the histogram and histogram_reshaped fiedls
        # check if it is already reshapded or not to avoid costly multiple reshape operations
        if(not reshaped):
            self.histogram = histogram
            self.histogram_reshaped = histogram.reshape(1,30,30,1)
        else:
            self.histogram_reshaped = histogram
            self.histogram = histogram.reshape(30,30)
            
                        
    def load_model(self, model):

        """ Function update

            Used the store the trained CNN model
            * Params: model
               - model : the specified model , must be a siamese network of (30,30) input shape        
        """
        # set the given model in the model field (Siamese Network of input_shape = (30,30))
        self.model = model
        # print('Model loadded')


    def identifie_cross(self, image,  keypoints_list, keypoints_coords,  lambda_factor = 0.8):
        
        """ Function identifie

            Identifie the person in the set of corrds given by PoseNet using both methods
            * Params: image, keypoint_list, lambda_factor
                - keypoint_list : should be be list of cross coords of the right shoulder and the 
                  left hip or the left shoulder and the right hip given by get_correct_coords
                - lambda_factor: the mixing factor between histogram correlation and Siamese Network 
                  confdences
        """
        # raise an exeption if person has not been initialized
        if (self.histogram is None):
            raise('This instance color has not been initialized')
        else:
            # dictionnary to stock the entire squeleton of person
            squeleton_dico = {}
            # list of confidences for comm comparaison with all squeletons
            confidences = []
            for squel, kp in zip(keypoints_coords,keypoints_list):
                
                if not (kp in None):

                    hist = perform_histogram(image, kp, Trace=False).reshape(1,30,30,1)

                    hist_reshaped = self.histogram_reshaped
                    # calculate the correlation distance between the person color histogram and the histogram given by kp
                    dist = cv2.compareHist(hist_reshaped, hist, cv2.HISTCMP_CORREL)
                    # calculate the One-Shot scor of correlation between the person histogram and the given histogram based on 
                    # the Siamese Network given
                    pred = self.model.predict([hist_reshaped, hist])
                    # For my trained model the One shot score is low for good matches
                    pred = 1 - pred[0,0]
                    # lambda_factor mixes the two confidences values (adapt our method on many cases)
                    # gives a better performance than the separates methods
                    conf = lambda_factor * pred + (1 - lambda_factor) * dist
                    # if the person is surely( conf>=0.9) identified, we updates his histogram
                    # this makes the histogram converge o the current histogram value independently on time
                    if(conf >= 0.9):
                        self.update_histogram(hist, reshaped = True)

                    confidences.append(conf)
                    squeleton_dico[conf] = squel

            return squeleton_dico, confidences
        