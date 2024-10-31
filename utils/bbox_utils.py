#to get center of bbox

def get_center(bbox):
    x1,y1,x2,y2 = bbox
    center_x = int((x1+x2)/2) 
    center_y = int((y1+y2)/2)
    return (center_x, center_y)

def measure_dist(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return (int((x1+x2)/2),y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_dist = float('inf')
    keypoint_ind = keypoint_indices[0]
    for keypoint_id in keypoint_indices:
        keypoint = keypoints[keypoint_id*2], keypoints[keypoint_id*2 + 1]
        dist = abs(point[1]-keypoint[1])
        
        if dist<closest_dist:
            closest_dist=dist
            keypoint_ind = keypoint_id
    return keypoint_ind

def get_height_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return y2-y1

def measure_xy_distance(p1,p2):
    return abs(p1[0]-p2[0]),abs(p1[1]-p2[1])