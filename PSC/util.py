import numpy as np
import cv2
#import confiance

###
def dist(P, Q=None):
    '''
    Function dist
    Calculate the distances.
    * Params: P, Q,   numpy.ndarray, of shapes (m, 2) and (n, 2)
        - If Q is None, Q will be valued as zeros_like(P).
        - If P, Q are simply 2 vectors, will return ||P-Q||.
        - If P, Q are two groups of vectors, will return the matrix D,
            such that D_ij = ||P_i - Q_j||.
    * Returns: 
        dist,  numpy.ndarray,  of shape (m,n). dist[i,j] = ||P[i] - Q[j]||.
    * Here the norm ||*|| is the quadratic norm.

    '''
    P = np.array(P, dtype=np.float64)
    if  isinstance(Q, np.ndarray): 
        Q = np.array(Q, dtype=np.float64)
    else:
        Q = np.zeros_like(P, dtype=np.float64)

    ## Shape check, wether these are vectors of the same dimension.
    if len(P.shape) > 2:
        raise Exception('P shall be a tensor of order 1 or 2.')
    if len(Q.shape) > 2:
        raise Exception('Q shall be a tensor of order 1 or 2.')
    if P.shape[-1] != Q.shape[-1]:
        raise Exception('The dimension of vector(s) in P does not match that of vector(s) in Q.')

    ## If P is simply a vectors.
    if len(P.shape) == 1:
        ## If Q is also a vector.
        if len(Q.shape) == 1:
            return np.linalg.norm(P-Q)
        ## If Q is not a vector but a matrix.
        else:
            return np.linalg.norm(P-Q, axis=1)
    ## If P is a list of vectors, but Q is just one vector.
    elif len(Q.shape) == 1:
        return np.linalg.norm(P-Q, axis=1)
    ## If both of them are lists of vectors.
    else:
        dim = P.shape[-1]
        Dist_2 = np.zeros((P.shape[0], Q.shape[0]))
        for i in range(dim):
            Qx, Px = np.meshgrid(Q[:,i], P[:, i])
            Dist_2 = Dist_2 + (Px-Qx)**2

        Dist = np.sqrt(Dist_2)
        return Dist




###
def match(coord_obj, coord_list_tar):
    '''
    Params: coord_obj, coord_list_tar
    Output: ind, conf
    Function used to find the corresponding target coord in the list $coord_list_tar 
    for the object coord $coord_obj.
    It returns the index of the corresponding coord, $ind,
    and the confiance of this macthing, $conf.

    '''
    coord_obj = np.array(coord_obj)
    coord_list_tar = np.array(coord_list_tar)

    ## Check the shapes.
    if len(coord_obj.shape) != 1:
        raise Exception('coord_obj shall be a vector.')

    if len(coord_list_tar.shape) != 2:
        raise Exception('coord_list_tar shall be a list of vectors, i.e. a matrix.')

    if coord_obj.shape[0] != coord_list_tar.shape[1]:
        raise Exception('The dimensions of coord_obj and coord_list_tar do not match.')

    ## threshold = 30

    ## Calculate the distances.
    dist_pr = dist(P=coord_list_tar, Q=coord_obj)
    ## Sort the distances
    #ind = np.argsort(dist_pr)

    ## The calculation of confiance needs the $cdf and $pdf of errs, there fore 
    ## we can not yet calculate it.

    # conf = np.zeros_like(ind)        
    # if dist_pr[ind[0]] < threshold:
    #     conf[0] = 1
    # else:
    #     conf[0] = 0.2

    # return ind[0], conf[0]

    ## We sort the indexes by distance,
    ## and calculate all the confiances.
    ind_l = np.argsort(dist_pr)
    conf_l = confiance.conf(dist_pr[ind_l])
    return ind_l, conf_l


###
def weighted_dist(coord_1, score_1, coord_2, score_2):
    '''
     Function: weighted_dist
            Calculate the weighted distance.

    '''
    coord_1 = np.array(coord_1)
    coord_2 = np.array(coord_2)

    coord_diff = coord_2 - coord_1
    dist_list = np.linalg.norm(coord_diff, axis=1)
    
    return np.dot(dist_list, score_1*score_2)



###
def to_tuple_coord(coord):
    '''
    Function : to_tuple_coord:
        Used to transform a posenet-type coord
        to the form that cv2 use.       
    Param(s) :
        coord   numpy.ndarray,  the coord to be processed. 
    Return(s):
        coord_cv2   numpy.ndarray, dtype=int, the coord transformed.

    '''
    coord = np.array(coord)
    coord = np.array(coord+0.5, dtype=int)
    return tuple(coord)[::-1]


## The list of lines that shall be draw.
Skeleton_pair_list = [[5,7], [7,9], [5,11], [6,12], [6,8], [8,10], [11,12], [11,13], [13,15], [12,14], [14,16]]
###
def draw_skeleton(img, coord_list, conf_list, min_score, color_p=(0,0,255), color_l=(255,0,0), width_offset = 0):
    '''
    Function : draw_skeleton:
        Used to draw the given skeletons in the given image,
        with specified point and line colors.
    * Param(s) :
        img,    numpy.ndarray, the image in which we draw.
        coord_list,     numpy.ndarray, (17,2), the list of points.
        conf_list,      numpy.ndarray, (17,1), the scores of points.
        min_score,       double,     the minimum score. 
        color_p,        tuple/color, bgr,     the color of points.
        color_l,        tuple/color, bgr,     the color of lines.  
    * Return(s) :
        img_,   numpy.ndarray, the image after drawing.

    '''
    coord_list = [to_tuple_coord(coord) for coord in coord_list]
    
    validity_list = [False]*len(coord_list)
    validity_list[0] = True
    
    img = cv2.circle(img, coord_list[0], 3, color_p, 3 + width_offset)
    
    ## Draw the points and verify their validities.
    for i in range(5, len(coord_list)):
        if conf_list[i] >= min_score:
            validity_list[i] = True
            img = cv2.circle(img, coord_list[i], 2, color_p, 2 + width_offset)
    
    ## Draw the shoulder and the head-line.
    if validity_list[5] and validity_list[6]:
        img = cv2.line(img, coord_list[5], coord_list[6], color_l, 2 + width_offset)
        coord_tmp = (np.array(coord_list[5]) + np.array(coord_list[6])) / 2
        coord_tmp = np.array(coord_tmp, dtype=int)
        coord_tmp = tuple(coord_tmp)
        img = cv2.line(img, coord_list[0], coord_tmp, color_l, 2 + width_offset)

    ## Draw all other pairs.
    for pair in Skeleton_pair_list:
        if validity_list[pair[0]] and validity_list[pair[1]]:
            img = cv2.line(img, coord_list[pair[0]], coord_list[pair[1]], color_l, 2 + width_offset)


    return img

