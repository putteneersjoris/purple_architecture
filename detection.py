import cv2
import numpy
from string import hexdigits

#create tracker object
import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        
        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            
            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[id] = (cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids


tracker = EuclideanDistTracker()


# node = hou.pwd()
# geo = node.geometry()


path = "./photoshop_files/a_1.jpg"


cap = cv2.VideoCapture(path, cv2.CAP_IMAGES)

# object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=5)
# object_detector = cv2.createBackgroundSubtractorKNN()
test = 0
while True:
    test +=1
    ret, frame = cap.read()

    height, width, _ = frame.shape
    # print (height, width)


    # extract image region
    roi = frame[0:3507,0:2480]
    

    # object detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ =  cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        # calc area and remove small elements
        
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(roi, [cnt], -1, (255,0,0),1)
            
            x,y,w,h = cv2.boundingRect(cnt)
            
            detections.append([x,y,w,h])

    # object tracking
    
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        
        x,y,w,h,id = box_id
        cv2.putText(roi, str(id), (x,y-15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
        cv2.rectangle(roi, (x,y), (x+w, y+h), (255,0,0),1)
        # print(box_id)
    
    cv2.imshow("roi",roi)
    cv2.imshow("Mask", mask)
    cv2.imwrite('./opencv_files/image_roi_' +str(test)+'.png',roi)
    cv2.imwrite('./opencv_files/image_mask_' +str(test)+'.png',mask)
    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()




