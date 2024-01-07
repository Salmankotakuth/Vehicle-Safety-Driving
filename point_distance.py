import cv2
import math
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import colors, Annotator


model = YOLO("yolov8n.pt")


names = model.model.names


cap = cv2.VideoCapture("Dashcam.mp4")


out = cv2.VideoWriter('dist_output.avi', cv2.VideoWriter_fourcc(*'MJPG'),30, (int(cap.get(3)), int(cap.get(4))))



reference_point_1 = (100, 1000)  # Set the reference point 1
reference_point_2 = (1800, 1000)  # Set the reference point 2



while True:
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break


    results = model.predict(frame, classes=[2, 5, 7]) # only detect car, truck and bus
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()


    # For drawing line on car's crash gaurd 
    cv2.line(frame, (0,1050), (int(reference_point_1[0]), int(reference_point_1[1])), (255, 0, 0), 4)
    cv2.line(frame, (int(reference_point_1[0]), int(reference_point_1[1])), (int(reference_point_2[0]), int(reference_point_2[1])), (255, 0, 0), 4)
    cv2.line(frame, (int(reference_point_2[0]), int(reference_point_2[1])), (2000, 1050), (255, 0, 0), 4)


    # Draw circle in the starting points
    cv2.circle(frame,(reference_point_1[0],reference_point_1[1]),10,(255,255,255),-1)
    cv2.circle(frame,(reference_point_2[0],reference_point_2[1]),10,(255,255,255),-1)


    cv2.putText(frame,('Car Dash Cam View'),(400,1040),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)


    for box, cls in zip(boxes, clss):


        cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(0,255,255),2)

        # calculate center
        cx=int(box[0]+box[2])//2
        cy=int(box[1]+box[3])//2

        # Plot center
        cv2.circle(frame,(cx,cy),10,(0,0,255),-1)

        # Calculate the distance
        object_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        # Calculate the distance from the reference point (10, 10)
        distance = math.sqrt((reference_point_1[0] - object_center[0]) ** 2 + (reference_point_1[1] - object_center[1]) ** 2)

        # Draw a line from the reference point to the object center
        cv2.line(frame, (reference_point_1[0],reference_point_1[1]), (int(object_center[0]), int(object_center[1])), (0, 255, 0), 5)

        # Display the distance on the image
        text = f"{distance:.2f} pixels"
        cv2.putText(frame, text, (int(object_center[0]), int(object_center[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)



        # Calculate the distance from the reference point (10, 10)
        distance = math.sqrt((reference_point_2[0] - object_center[0]) ** 2 + (reference_point_2[1] - object_center[1]) ** 2)

        # Draw a line from the reference point to the object center
        cv2.line(frame, (reference_point_2[0],reference_point_2[1]), (int(object_center[0]), int(object_center[1])), (255, 255, 0), 5)

        # Display the distance on the image
        text = f"{distance:.2f} pixels"
        cv2.putText(frame, text, (int(object_center[0]), int(object_center[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)



    


    out.write(frame)


    resized_frame = cv2.resize(frame, (1020, 550))


    cv2.imshow("vehicle distance calculation", resized_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


out.release()
cap.release()
cv2.destroyAllWindows()