###########
# This is the final version of our tracking project.


import numpy as np
import cv2
from posenet.constants import PART_IDS as PID
TARGETED_PARTS_IDS = [PID["rightShoulder"], PID["leftHip"], PID["leftShoulder"], PID["rightHip"]]

from Person import Person
from functions import get_key_point_list, link_confidences


class Track:
    '''
    The class used to identify, track and draw the skeletons.
    '''
    ####
    # Global Variables:
    min_pose_score = 0.05   
    
    # Colors in code BGR, red, blue, green and yellow.
    COLOR_LIST = [(0,0,255), (255,0,0), (0,255,0), (0,255,255)]
    NAME_LIST = ['Red', 'Blue', 'Green', 'Yellow']


    ####
    # Functions:
    def __init__(self, num_skeleton, max_sample_size=10, sample_size=7, shape=(2,), deg=3, min_score=0.2, min_conf=0.3):
        self.num_skeleton = num_skeleton
        self.max_sample_size = max_sample_size
        self.sample_size = sample_size

        self.min_score = min_score
        self.min_conf = min_conf

        # Init the person list.
        self.person_list = []           # The list of all persons.
        self.valid_person_list = []     # The list of valid persons.
        for i in range(num_skeleton):
            person = Person(max_sample_size, sample_size, shape, deg, min_score, min_conf)
            # We may add other attributes like name, etc.
            self.person_list.append(person)

        # Allocate colors.
        if self.num_skeleton > 4:  # If we have more than 4 person.
            phases = np.arange(self.num_skeleton, dtype=np.uint8)*180/self.num_skeleton
            colors = np.zeros(self.num_skeleton, 3, dtype=np.uint8) # The HSV colors.
            colors[:,0] = phases
            colors[:,1] = 255*np.ones_like(colors[:,1])
            colors[:,2] = 255*np.ones_like(colors[:,2])
            self.COLOR_LIST = cv2.cvtColor(colors, HSV2BGR)

        for i in range(num_skeleton):
            self.person_list[i].color_point = self.COLOR_LIST[i]
            self.person_list[i].color_line = self.COLOR_LIST[i]
            self.person_list[i].name = self.NAME_LIST[i]

        self.num_data = 0

        self.frame_color = 0 # for statistics

    def initialize(self, new_coord_list, new_score_list, frame=None):
        '''
        When we start a track process, this function is used to initialise
        all the information, like the initial coords, color features, and etc.
        The given coords, scores are supposed to sorted corresponding to the 
        persons.
        * Param(s):
            new_coord_list,     numpy.ndarray, the list of new coords.
            new_score_list,     numpy.ndarray, the list of new scores.
        '''
        
        # Check the number of coords, scores and pose-scores.
        if not (    len(new_coord_list) == self.num_skeleton\
                and len(new_score_list) == self.num_skeleton):
            raise('Error, the sizes of coord_list, score_list, or pose-score_list\
                do not match up with the num_skeleton.')
        
        # Store the infos in each person.
        for i in range(self.num_skeleton):
            self.person_list[i].update(new_coord_list[i], new_score_list[i])

        # After the initialization, all the persons shall be valid.
        self.valid_person_list = self.person_list
        self.num_data += 1
        
        # Initialize the histograms for color distinction
        if not (frame is None):
            keypoint_coords, keypoint_list = get_key_point_list(new_coord_list, TARGETED_PARTS_IDS)
            #print(keypoint_list)
            for kp, person in zip(keypoint_list, self.person_list):
                person.put_histogram(frame, kp)


    def update(self, new_coord_list, new_score_list, new_pose_score_list, frame):
        '''
        Used to identify and track the skeletons, and to upgrade the infomations.
        * Param(s):
            new_coord_list,     numpy.ndarray, the list of new coords.
            new_score_list,     numpy.ndarray, the list of new scores.
            new_pose_score_list, numpy.ndarray, the list of new pose socres.

        '''
        # Filter to get the valid coords and scores.
        vld_coord_list, vld_score_list = self.coord_filter(new_coord_list, new_score_list, new_pose_score_list)
        num_vld_coord = len(vld_coord_list)
        num_vld_person = len(self.valid_person_list)
        # The list of tuples containing the coord-score pair.
        cs_tup_list = [(vld_coord_list[i], vld_score_list[i]) for i in range(num_vld_coord)]

        candid_list = self.person_list.copy()
        valid_person_list_new = []  # The list of people that are valid after this update.
        non_match_list = []     # The list of people that do not have matching.
        for person in self.person_list:
            if not person.validity :
                non_match_list.append(person) 

        print('**** Pre start *******')
        cs_tup_list_copy = cs_tup_list.copy()
        match_coord_ind = []
        for i in range(len(cs_tup_list)):
            cs_tup = cs_tup_list[i]
            print(125, len(cs_tup_list))
            C = [] # store the confidence of each person for this coordinate
            for person in self.person_list:
                if person.validity:
                    C.append(person.corr_score(cs_tup[0], cs_tup[1]))
            if len(C) > 0:
                C = np.array(C)
                ind_max = np.argmax(C)
                conf_max = C[ind_max]
                person_match = self.person_list[ind_max]
                C[ind_max] = 0
                if ( (conf_max-C)>0.5 ).all():
                    print(139, person_match.name)
                    person_match.update(cs_tup[0], cs_tup[1])
                    valid_person_list_new.append(person_match)
                    if person_match in candid_list:
                        candid_list.remove(person_match)
                    match_coord_ind.append(i)
        cs_tup_list = []
        for i in range(len(cs_tup_list_copy)):
            if not (i in match_coord_ind):
                cs_tup_list.append(cs_tup_list_copy[i])

        # After the loop above, we have:
        #   - cs_tup_list is empty (i.e. all coords are used),
        # Or- candid_list is empty (i.e. no person can be matched)
        # cs_tup_list : [(coord, score)]
        # non_match_list : [person]
        if len(cs_tup_list) > 0:
            #**************
            # HERE WE USE THE COLOR / FACE DETECTION 
            #**************
            self.frame_color += 1

            # COLOR if color can be use
            num_person = len(candid_list)
            num_tup_list = len(cs_tup_list)
            # if we have more person than given squeletons, we attribute a None element to these we don't have and 0 score
            if(num_tup_list<num_person):
                for i in range(num_person - num_tup_list):
                    cs_tup_list.append((None, 0))
            
            # create the list containing the keypoints of every person
            keypoint_coords = [tuple[0] for tuple in cs_tup_list]
            # These two lists contains the same person keypoints at every index and None for those are None or don't have targeted parts
            
            keypoint_coords , keypoint_list = get_key_point_list(keypoint_coords, TARGETED_PARTS_IDS)

            # we will store in the dictionnaries the complete skeleton of the person if identified
            dico_list = []
            # The list of ths confindence lists containing the match value of person with all the keypoints
            confidence_list = []
            for person in candid_list:
                #print(person.name)
                # frame is the current frame of the tracker: must be outside the class ?
                # person.identifie_cross -> dico,confidences_list
                    # dico : keys = conf or -10  , values : sqeleton whose histogram correlation with person.histogram is conf or None for -10 as key
                dico_confidence = person.identifie_cross(frame, keypoint_list, keypoint_coords, lambda_factor=0.8)
                # 0.8 is a good value for lambda_factor
                dico_list.append(dico_confidence[0])
                confidence_list.append(dico_confidence[1])
                
            # match_confidence_list contains the value for the match  
            # gives the list of maximum cofidences of each person on the keypoints while avoiding conflicts
            match_confidence_list = link_confidences(confidence_list)  
            
            for i, person in enumerate(candid_list):
                # print(person.name)
                conf_matched = match_confidence_list[i]
                # dico_list[i] -> dictionnary with keys; confidence , values; sqeleton of this calculated confidence relative to person_i
                # dico_list[i][conf_matched] -> squeleton of the maximum correlation 
                #person.update(dico_list[i][conf_matched], conf_matched)
                new_coord = dico_list[i][conf_matched]
                if new_coord is None:
                    person.update(None, None)
                else:
                    for cs_tup in cs_tup_list:
                        if np.isclose(cs_tup[0], new_coord).all():
                            #print(cs_tup[0])
                            person.update(cs_tup[0], cs_tup[1])
                            # non_match_list.remove(person)
                            break
                #person.update()
           
            
           # FACE if face can be use
           #TODO

        # Now no coord is remaining.
        # for person in non_match_list:
        #     person.validity = False
        # # Update the valid_person_list
        self.valid_person_list = valid_person_list_new

        





    def coord_filter(self, new_coord_list, new_score_list, new_pose_score_list):
        '''
        Used to filter out the valid coords.
        * Param(s):
            new_coord_list,     numpy.ndarray, the list of new coords.
            new_score_list,     numpy.ndarray, the list of new scores.
        * Return(s):
            vld_coord_list,     numpy.ndarray, the list of valid coords.
            vld_score_list,     numpy.ndarray, the list of valid score.
        '''
        vld_coord_list = []
        vld_score_list = []

        for i in range(len(new_coord_list)):
            if new_pose_score_list[i] >= self.min_pose_score:
                vld_coord_list.append(new_coord_list[i])
                vld_score_list.append(new_score_list[i])
        
        vld_coord_list = np.array(vld_coord_list)
        vld_score_list = np.array(vld_score_list)

        return vld_coord_list, vld_score_list



    def init_match(self, person_list, cs_tup_list):
        '''
        Used to calculate the initial matching list.
        * Param(s):
            person_list, list(Person), the list of people.
            cs_tup_list, list(tuple), the list of coord-score pairs.
        * Return(s):
            init_corr_tup_list, list(tuple), each tuple is (coord, score, corr_list). 
                init_corr_list[i] is the tuple (coord, score, corr_list),
                where corr_list is the list of all people that are corresponding 
                initially to the coord.
        '''
        # build the coord and score lists.
        coord_list = []
        score_list = []
        for tup in cs_tup_list:
            coord_list.append(tup[0])
            score_list.append(tup[1])
        coord_list = np.array(coord_list)
        score_list = np.array(score_list)

        length = len(coord_list)    # Length of coords.
        tar_list = coord_list[:,0]  # The list of targets.
        # Initialise as empty.
        init_corr_list = []
        for i in range(length):
            init_corr_list.append([])   # Each list is initially empty.
        
        for person in person_list:
            ind_list, conf_list = person.match(tar_list)
            # If the init corr coord for person P is C,
            # we add P to the init_corr_list of C.
            init_corr_list[ind_list[0]].append(person)  

        # Build the init_corr_tup_list.
        init_corr_tup_list = []
        for i in range(length):
            coord = coord_list[i]
            score = score_list[i]
            corr_list = init_corr_list[i]
            init_corr_tup_list.append((coord, score, corr_list))

        return init_corr_tup_list

