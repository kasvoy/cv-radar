from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
from speed_funcs import assign_point_time, get_point_under_transform, TRANSFORM_MATRIX

model = YOLO('yolov8l.pt')
video_path = "data/session5_center/video.avi"

box_ann = sv.BoxAnnotator(thickness=2, text_scale=0.5, text_thickness=1)


def main():
    frame_number=0
    
    for result in model.track(source=video_path, show=False, stream=True, tracker='bytetrack.yaml', classes=[1, 2, 3, 5, 6, 7, 8]):
            
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)
        boxes = result.boxes.xyxy.cpu().numpy()
        
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            detections.speeds = list(range(len(detections.tracker_id)))
            track_tuples = (list(zip(detections.tracker_id, boxes)))
            
        print(track_tuples)
        
        for track in track_tuples:
            track_id = track[0]
            bbox_coords = track[1]
            
            br = np.array([int(bbox_coords[1]), int(bbox_coords[2])], dtype=np.float32)
            br_transformed = get_point_under_transform(br, TRANSFORM_MATRIX)
        
        labels = [
            f"{tracker_id} {model.model.names[class_id]} {confidence:0.2f} speed: {speeds}"
            for _, _, confidence, class_id, tracker_id, speeds
            in detections
        ]

        frame = box_ann.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )


        cv2.imshow("yolov8", frame)
        frame_number += 1
        
        key = cv2.waitKey(1)
                
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('p'):
            cv2.waitKey(-1)


if __name__ == '__main__':
    main()
