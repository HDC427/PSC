import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import segmentation
from bokeh.plotting import figure, output_file, show


def mean_image(image, label):

    """ Function mean_image

        Transform a segmented Tensor by K_means into an Image segmented
        * Params: image, label
           - Contains the different labels from the segmentation
        * Returns: the Tensor of the input image
    """

    im_rp = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
    sli_1d = np.reshape(label, -1)
    uni = np.unique(sli_1d)
    uu = np.zeros(im_rp.shape)
    for i in uni:
        loc = np.where(sli_1d==i)[0]
        mm=np.mean(im_rp[loc,:], axis=0)
        uu[loc,:] = mm
    oo = np.reshape(uu, [image.shape[0], image.shape[1], image.shape[2]]).astype(np.uint8)
    return oo


def contains_pers(image, Persons_list, coords, crit):
    """ Function contains_pers

        Check if the Person wich coords are given by coords are or not in Person_list
        with the confidence criteria crit
        * Params: image, Person_list, coords, crit
           - Perform the histogram and compare it to the others
           - Retutrn true if one of them realises a distance superior to crit
        * Returns: boolean
    """

    histogram = perform_histogram(image, coords, Trace=False)
    contain = False

    for person in Persons_list:
        hist1 = person.Color.histogram
        if(not (hist1 is None)):
            distance = cv.compareHist(histogram, person.Color.histogram, cv.HISTCMP_CORREL)
            print(distance)
            if distance > crit:
                contain = True
                break
    return contain


def get_key_point_list(keypoint_coords, TARGETED_PART_IDS):

    """ Function get_correct_coords

        From the results of Keypoints and Scores given by PosNet, return the keypoints for 
        the ROI to extract the color of the cloths of the num_pers persons
        * Params: pose_scores, keypoint_coords, keypoint_coords, TARGETED_PART_IDS, num_pers
           - TARGETED_PART_IDS contains the crossed shoulders and hips ids in PoseNet
        * Returns: Person_keypoints 
            - list of keypoints for each person
    """

    keypoint_list = []
    coords = []

    for i, keypoint in enumerate(keypoint_coords):
        if keypoint is None:
            keypoint_list.append(None)
        else:
            coords_0 = []
            coords_1 = []

            for ii, kc in enumerate(keypoint):
                if ii in TARGETED_PART_IDS[:2]:
                    coords_0.append(kc)
                if ii in TARGETED_PART_IDS[2:]:
                    coords_1.append(kc)

            if(len(coords_0) == 2):
                coords = coords_0
            else:
                coords = coords_1

            if(len(coords) == 2):
                keypoint_list.append(coords)
            else:
                keypoint_list.append(None)

    return keypoint_coords, keypoint_list



def get_correct_coords(pose_scores, keypoint_scores, keypoint_coords,TARGETED_PART_IDS,
                        num_pers, min_pose_confidence=0.5, min_part_confidence=0.5):

    """ Function get_correct_coords

        From the results of Keypoints and Scores given by PosNet, return the keypoints for 
        the ROI to extract the color of the cloths of the num_pers persons
        * Params: pose_scores, keypoint_coords, keypoint_coords, TARGETED_PART_IDS, num_pers
           - TARGETED_PART_IDS contains the crossed shoulders and hips ids in PoseNet
        * Returns: Person_keypoints 
            - list of keypoints for each person
    """

    keypoint_list = []

    for ii, score in enumerate(pose_scores):
        coords_0 = []
        coords_1 = []
        if score < min_pose_confidence:
            continue

        for iii, (ks, kc) in enumerate(zip(keypoint_scores[ii,:], keypoint_coords[ii, :, :])):
            if ks > min_part_confidence and (iii in TARGETED_PART_IDS[:2]):
                coords_0.append(kc)
            if ks > min_part_confidence and (iii in TARGETED_PART_IDS[2:]):
                coords_1.append(kc)
        if(len(coords_0)==2):
            coords = coords_0
        else:
            coords = coords_1

        if(len(coords) == 2):
            keypoint_list.append(coords)

    return keypoint_list


def segmentation(ROI ,compactness=50, n_segments=50):

    """ Function segmentation

        Segment the image into n_segments segments using K_means
        * Params: ROI, compactness, n_segments
           - ROI the Region Of Interest
        * Returns: output 
            - Segmented image
    """

    roi = cv.cvtColor(ROI, cv.COLOR_BGR2RGB)
    label = segmentation.slic(roi, compactness=compactness, n_segments=n_segments)
    output = mean_image(r,label)
    return output


def plot_test(liste_points):

    """ Function plot_test

        Show a figure of the results
        * Params: liste_points
        * Returns: void
    """
    p = figure(title="probability", plot_width=300, plot_height=500)
    p.line(x= range(len(liste_points)), y=liste_points)
    show(p)

    
def perform_histogram(frame, keypoints, Trace=True):

    """ Function perform_histogram
        
        Create the histogram of Colorof the cloth
        * Params: frame, parameter, Trace
           - parameter : coords from get_correct_coords
           - Trace : for results
        * Returns: hist
    """

    A = [0,1,2,3]
    A[0] , A[1] = int(min(keypoints[0][0],keypoints[1][0])) , int(max(keypoints[0][0],keypoints[1][0]))
    A[2] , A[3] = int(min(keypoints[0][1],keypoints[1][1])) , int(max(keypoints[0][1],keypoints[1][1]))
        
    region = frame[A[0]-6:A[1]+6,A[2]-2:A[3]+2,:]
    #output = sharpening(region, intensity=1)
    output = region    
    if Trace:
        cv.imshow('frame',output)
        cv.waitKey()

    hsv = cv.cvtColor(output,cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv],[0,1],None,[30,30],[0,180,0,255]) 
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    hist = hist.astype(np.float32) / 255

    if Trace:
        plt.matshow(hist,extent=[0,255,0,180],cmap=plt.cm.gray)
        plt.ylabel("Hue")
        plt.xlabel("Sat")
        plt.savefig('histogram.png')
        #plt.imsave('histogram.png',hist)
        plt.show()
        cv.destroyAllWindows()

    return hist

def sharpening(ROI, intensity=1):

    """ Function sharpening

        Sharpen the image using convolution with different intensities
        * Params: ROI, intensity
           - ROI the Region Of Interest
           - Intensity
               0 : Low Sharpening
               1 : Medium Sharpening
               2 : Strong Sharpening
    * Returns: region 
           - Sharpened image
    """

    if intensity==0:
        kernel = np.array([[1,1,1],[1,-7,1],[1,1,1]])          ## Low Sharpening
    if intensity==1:
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])   ## medium Sharpening
    if intensity==2:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])       ## Strong sharpening
    region = ROI
    region = cv.filter2D(ROI,-1,kernel)
    region = cv.medianBlur(region, 5)
    return region

def perform_histogramBGR(frame, keypoints, Trace=True):

    """ Function perform_histogramBGR
        
        Create the coloredqhistogram of Color of the cloth in BGR
        * Params: frame, parameter, Trace
           - parameter : coords from get_correct_coords
           - Trace : for screening results
        * Returns: hist
    """

    A = [0,1,2,3]
    A[0] , A[1] = int(min(keypoints[0][0],keypoints[1][0])) , int(max(keypoints[0][0],keypoints[1][0]))
    A[2] , A[3] = int(min(keypoints[0][1],keypoints[1][1])) , int(max(keypoints[0][1],keypoints[1][1]))
        
    region = frame[A[0]-5:A[1]+5,A[2]-5:A[3]+5]
    output = sharpening(region, intensity=1)
    #output = region    
    if Trace:
        cv.imshow('frame',output)
        cv.waitKey()

    #hsv = cv.cvtColor(output,cv.COLOR_BGR2HSV)
    hsv = output
    hist = cv.calcHist([hsv],[0,1,2],None,[50,50,50],[0,250,0,255,0,250]) 
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        
    if Trace:
        plt.matshow(hist,extent=[0,255,0,250,0,255],cmap=plt.cm.gray)
        plt.ylabel("Hue")
        plt.xlabel("Sat")
        plt.imsave('histogram.jpg',hist)
        plt.show()
        cv.destroyAllWindows()

    return output


def link_confidences_2(confidences_0, confidences_1):

    """ Function link_confidences
        
        Binarizes the identification task using pedictions
        * Params: confidences_0 , confidences_1
           - confidences_0 : list of confidences returned for the first person
           - confidences_1 : list of confidences returned for the second person
           - Trace : for results
        * Returns: conf_0 , conf_1
    """

    if(len(confidences_0) == 1):
        return confidences_0[0] , confidences_1[0]

    conf_0 , conf_1 = max(confidences_0) , max(confidences_1)
    confidences = np.array([confidences_0, confidences_1])

    if (conf_0 in confidences[:,0]) and (conf_1 in confidences[:,1]) or (conf_0 in confidences[:,1]) and (conf_1 in confidences[:,0]):
        
        return conf_0, conf_1

    else:
        if (conf_0 in confidences[:,0]) and (conf_1 in confidences[:,0]):

            if(conf_0 >= conf_1):
                conf_1 = confidences[1,1]
            else:
                conf_0 = confidences[0,1]

        if (conf_0 in confidences[:,1]) and (conf_1 in confidences[:,1]):

            if(conf_0 >= conf_1):
                conf_1 = confidences[1,0]
            else:
                conf_0 = confidences[0,0]

        return conf_0, conf_1
    
        
def link_confidences(confidences_list):

    """ Function link_confidences
        
        Extract the best confidances repartition from the list of confidances list 
        * Params: confidences_list
           - confidences_list : list of confidences_list returned for each person
        * Returns: linked_confidances
    """
    
    if(len(confidences_list) == 0):
        raise('confidences_list is empty') 
    if(len(confidences_list) == 1):
        if(len(confidesnces_list[0]) == 1):
            return [confidences_list[0][0]]
        else:
            return []
        
    # All the lists in confidences list has the same length
    confidence_0 = confidences_list[0]
    confidence_1 = confidences_list[1]
    if(len(confidence_0) == 0):
        return []
    if(len(confidence_0) == 1):
        return [confidence_0[0] , confidence_1[0]]
    
    
    else:
        # Dancing links structure for more than 2 unknown
        # n = len(confidences_list)
        confidences = np.array(confidences_list) 
        n = len(confidences)
        linked_confidances = [-10 for id in range(n)]
        columns = [c for c in range(n)]
        rows = [r for r in range(n)]
        
        while(confidences.size != 0):
            print('size : ',confidences.size)
            max_index_list = []
            element_to_delete = []
            for row in confidences:
                # (i,maw_index_list[i]) for i<=n arethe maximux confidences
                max_index_list.append(np.argmax(row))
               
            n = len(max_index_list)
            for i in range(n):
                if(max_index_list.count(i) == 1):
                    index_i = max_index_list.index(i)
                    linked_confidances[rows[index_i]] = confidences[index_i,i]
                    element_to_delete.append((index_i,i))
                    # confidences = np.delete(confidances, index_i,0)
                    # confidences = np.delete(confidances, i,1)
                    
                if(max_index_list.count(i) > 1):
                    maximums = [confidences[ii,i] if max_index_list[ii] == i else -15 for ii in range(n)]
                    max_id = np.argmax(maximums)
                    linked_confidances[rows[max_id]] = maximums[max_id]
                    element_to_delete.append((max_id,i))
                    # confidences = np.delete(max_id,0)
                    # confidences = np.delete(i,1)
            print(linked_confidances)       
            # i_prev,j_prev = n,n
            n_rows_deleted_above,n_columns_deleted = 0,0
            deleted_rows = []
            for element in element_to_delete:
                i,j = element
                #if(i>=i_prev):
                print('Before', i , j)
                n_rows_deleted_above = len([deleted_row for deleted_row in deleted_rows if deleted_row<i])
                i -= n_rows_deleted_above
                j -= n_columns_deleted
                print(i,j)
                confidences = np.delete(confidences,i,0)
                deleted_rows.append(i)
                confidences = np.delete(confidences,j,1)
                n_columns_deleted += 1
                rows.pop(i)
                columns.pop(j)
                i_prev,j_prev = i,j
        
        return linked_confidances
      
