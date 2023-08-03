import cv2
import numpy as np

"""
br = [1598, 967]
bl = [470, 924]
tl = [911, 85]
tr = [1273, 85]

br_new = [1400, 959]
tr_new = [1400, 60]
tl_new = [1000 ,60]
bl_new = [1000, 959]

src = np.float32([tl, tr, br, bl])
dst = np.float32([tl_new, tr_new, br_new, bl_new])

TRANSFORM_MATRIX = cv2.getPerspectiveTransform(src, dst)
"""

TRANSFORM_MATRIX = np.array([[     1.5617,      4.1251,     -435.51],
                            [  0.0035754,      4.6042,     -314.34],
                            [  5.959e-05,   0.0033356,           1]])

#ROI includes only the lanes going in direction of the camera
def extract_roi(frame):
    return frame[:, :1450] 

def get_point_under_transform(pt, transform_matrix):
    #point has to be an np.array, dtype=np.float32
    return cv2.perspectiveTransform(pt.reshape(-1, 1, 2), transform_matrix).reshape(2,)


#add an entry to the dictionary which cars crossed which points at what times - changes state of dictionary
def assign_point_time(speeds_dict, points, br_transformed, track_id, time, tolerance):
    
    for point in points.items():
        
        point_name = point[0]
        point_y_coord = point[1][1]
        previous_point_name = 'P'+str(int(point_name[1])-1)
        offset = 0
        first_time_here = False    
        
        if point_name == 'P1':
            offset = 10
            
        if br_transformed[1] >= point_y_coord-(tolerance+offset) and br_transformed[1] <= point_y_coord+tolerance:
            if speeds_dict[track_id][point_name] == 0:
                speeds_dict[track_id][point_name] = time
                first_time_here = True
            
            if point_name != 'P1' and first_time_here:
                time_at_prev = speeds_dict[track_id][previous_point_name]
                
                if time_at_prev != 0:
                    delta = round(time - time_at_prev, 3)
                    section_speed = round((7/delta)*3.6, 3)
                    
                    if previous_point_name == 'P1':
                        speeds_dict[track_id]['delta12'] = delta
                        speeds_dict[track_id]['speed12'] = section_speed
                                    
                    elif previous_point_name == 'P2':
                        speeds_dict[track_id]['delta23'] = delta
                        speeds_dict[track_id]['speed23'] = section_speed
                                            
                    elif previous_point_name == 'P3':
                        speeds_dict[track_id]['delta34'] = delta
                        speeds_dict[track_id]['speed34'] = section_speed

                    else:
                        speeds_dict[track_id]['delta45'] = delta 
                        speeds_dict[track_id]['speed45'] = section_speed

def assign_point_frames(speeds_dict, points, br_transformed, track_id, frame_number, tolerance):
    for point in points.items():
            
        point_name = point[0]
        point_y_coord = point[1][1]
        previous_point_name = 'P'+str(int(point_name[1])-1)
        offset = 0
        first_time_here = False    
        
        if point_name == 'P1':
            offset = 10
        
        if br_transformed[1] >= point_y_coord-(tolerance+offset) and br_transformed[1] <= point_y_coord+tolerance:
            if speeds_dict[track_id][point_name] == 0:
                speeds_dict[track_id][point_name] = frame_number
                first_time_here = True
                
        if point_name != 'P1' and first_time_here:
            fnumber_at_prev = speeds_dict[track_id][previous_point_name]
            
            if fnumber_at_prev != 0:
                frame_delta = frame_number - fnumber_at_prev
                time_delta = frame_delta/50
                section_speed = round((7/time_delta)*3.6, 3)
                
                if previous_point_name == 'P1':
                    speeds_dict[track_id]['delta12'] = time_delta
                    speeds_dict[track_id]['speed12'] = section_speed
                                    
                elif previous_point_name == 'P2':
                    speeds_dict[track_id]['delta23'] = time_delta
                    speeds_dict[track_id]['speed23'] = section_speed
                                        
                elif previous_point_name == 'P3':
                    speeds_dict[track_id]['delta34'] = time_delta
                    speeds_dict[track_id]['speed34'] = section_speed

                else:
                    speeds_dict[track_id]['delta45'] = time_delta 
                    speeds_dict[track_id]['speed45'] = section_speed


def display_video_time(current_time, frame):
    current_time_trunc = float('{:.2f}'.format(current_time))

    text = f"Video time: {current_time_trunc}"

    cv2.putText(frame, text, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 2)