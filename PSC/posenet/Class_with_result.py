import cv2

import torch
import argparse
from random import randint as rand
import numpy as np

import posenet
from functions import *

#### Regler le probleme des 4 coordonnees ####

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.5) ## default=0.7125
parser.add_argument('--delta', type=int, default=25)
args = parser.parse_args()

class Person:

    def __init__(self, First_Name, Second_Name, image, coords):

        ### Ajouter des parametres pour identifier la personne ###

        self.First_Name = First_Name 
        self.Second_Name = Second_Name
        self.color = (rand(0,255), rand(0,255), rand(0,255))
        
        right_shoulder = coords[0]
        left_hip = coords[1]
        y_min , y_max = int(min(right_shoulder[0],left_hip[0])) , int(max(right_shoulder[0],left_hip[0]))
        x_min , x_max = int(min(right_shoulder[1],left_hip[1])) , int(max(right_shoulder[1],left_hip[1]))

        self.histogram = perform_histogram(image, y_min, y_max, x_min, x_max)

        self.prob =[]

class Unknown_Person():

    def __init__(self):

        self.First_Name = "Unknown"
        self.Second_Name = "Person"

        self.color = (0,255,0)


class Thresh_Identifier: ###Change les  0.15 par 0.1

    PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    ]


    def __init__(self, videocapture, num_pers=4, min_score=0.1):
    
        self.cap = videocapture
        self.cap.set(3, args.cam_width)
        self.cap.set(4, args.cam_height)

        self.num_pers = num_pers
        self.min_score = min_score

        self.Persons = []

        TARGETED_PART_IDS = [Thresh_Identifier.PART_NAMES.index(PART) for PART in ["rightShoulder", "leftHip",
                                                                                "rightHip","leftShoulder"]]

        model = posenet.load_model(args.model)
        #model = model.cuda()
        output_stride = model.output_stride

        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                self.cap, scale_factor=args.scale_factor, output_stride=output_stride)

            with torch.no_grad():
                input_image = torch.Tensor(input_image)#.cuda()
    
                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=4,   ###
                    min_pose_score=0.15)      ### 

            keypoint_coords *= output_scale

            Pers_Keypoints = get_correct_coords(pose_scores, keypoint_scores, keypoint_coords,
                                                TARGETED_PART_IDS, self.num_pers,
                                                min_pose_confidence=0.15, min_part_confidence=0.1)
            
            #num_pers = len(pose_scores) - list(pose_scores).count(0.0)

            for i, coords in enumerate(Pers_Keypoints):
                if not contains_pers(display_image, self.Persons, coords, crit=0.80) and i<self.num_pers:

                    self.Persons.append(Person('Personne', str(len(self.Persons)+1), display_image, coords))

            if(len(self.Persons) == self.num_pers):
#                print(Pers_Keypoints)  ####
                break
        
        print("Initialisation succed")
#        self.frame = display_image
#
#        for i,coords in enumerate(Pers_Keypoints):
#            self.Persons.append(Person('Personne', str(i), self.frame, coords))


    def _Identifie(self, image, right_shoulder, left_hip):

        y_min , y_max = int(min(right_shoulder[0],left_hip[0])) , int(max(right_shoulder[0],left_hip[0]))
        x_min , x_max = int(min(right_shoulder[1],left_hip[1])) , int(max(right_shoulder[1],left_hip[1]))

        histogram = perform_histogram(image, y_min, y_max, x_min, x_max, Trace=False)

        Person = Unknown_Person()

        Dico_dist = {}
        distances = []

        for person in self.Persons:
            dist = cv2.compareHist(histogram, person.histogram, cv2.HISTCMP_CORREL) #cv2.HISTCMP_CHISQR)#
                    #    print(dist)
        #    if dist>0.8:
        #        Person = person
        #        print("yes")

            Dico_dist[dist] = person
            distances.append(dist)

#        dist = max(distances)
        dist = max(distances)

        print(dist)
        Person = Dico_dist[dist]
        Dico_dist[dist].prob.append([dist,int(Person.Second_Name)])              # """"""""""""""""""""""""""""""""""""""""""""

#        if dist>0.5:
#            Person = Dico_dist[dist] ## Method CV_HISTMAP_CORREL
#        color = ((ord(Person.First_Name[0])*1000)%255,0,(ord(Person.Second_Name[-1])*1000)%255)

        d = args.delta
        cv2.rectangle(image, (x_min-d,y_min-d), (x_max+d,y_max+d), color=Person.color, thickness=6)
        cv2.putText(image,'{} {} \n Prob = {}'.format(Person.First_Name, Person.Second_Name, dist), 
                    (x_min,y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=2)



if __name__ == '__main__':
    path = r'C:\Users\TRMoussa-PCHP\Desktop\PSC\Detection_Color\Videos\Duo_1_Trim.mp4'
    I = Thresh_Identifier(cv2.VideoCapture(path), num_pers=2)
    #exit()
    #print('Initialisation succed')
    TARGETED_PART_IDS = [I.PART_NAMES.index(PART) for PART in ["rightShoulder", "leftHip",
                                                                "rightHip","leftShoulder"]]

    model = posenet.load_model(args.model)
        #model = model.cuda()
    output_stride = model.output_stride
    i=0
    while True:

        input_image, display_image, output_scale = posenet.read_cap(
                I.cap, scale_factor=args.scale_factor, output_stride=output_stride)

        #input_image = cv2.resize(input_image, (0,0),None,0.4,0.4)
        #display_image = cv2.resize(display_image, (0,0),None,0.4,0.4)


        with torch.no_grad():
            input_image = torch.Tensor(input_image)#.cuda()
    
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        Pers_Keypoints = get_correct_coords(pose_scores, keypoint_scores, keypoint_coords,
                                            TARGETED_PART_IDS, I.num_pers,
                                            min_pose_confidence=0.15, min_part_confidence=0.1)
        if len(Pers_Keypoints) != 0:

            for coords in Pers_Keypoints:              ############################################
                I._Identifie(display_image, coords[0], coords[1])

#        cv2.imshow('frame',display_image)
#        k =cv2.waitKey(1)
        #if k == ord('q'):
        #    break
        i += 1
        if(i==500):
            break

    for pers in I.Persons:
        mat = np.array(pers.prob)
        np.savetxt('mat'+str(pers.Second_Name)+'.csv',mat)

