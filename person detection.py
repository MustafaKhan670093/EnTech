import cv2
from imageai.Detection import ObjectDetection
import os
from clarifai import rest
from clarifai.rest import ClarifaiApp


#Returns a working directory for the actual folder of the file.
execution_path = os.getcwd()

print(execution_path)

#Initialize the detector.
detector = ObjectDetection()

#This sets the initial object detection model instance to the pre trained "RetinaNet" model. 
detector.setModelTypeAsRetinaNet()

#Set the model path of the model file we downloaded (the resnet model that uses the COCO database)
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))

#Load the model.
detector.loadModel()
cap = cv2.VideoCapture(1) 
#Note: 0 is internal cam and 1 is external webcam.

z=0
while(1):
    #Photo analysis.
    z += 1
    ret, frame = cap.read()
    cv2.imshow("imshow",frame)
    cv2.waitKey(1)
    print(z)
    if z == 30:
        z = 0
        y = 0
        cv2.imwrite('C:/Users/adity/desktop/image'+str(y)+'.png', frame)
        print("Wrote Image")

        detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, 'C:/Users/adity/desktop/image'+str(y)+'.png'),
                                                     output_image_path=os.path.join(execution_path, "output.jpg"))
        for x in detections:
            if(x["name"] == "person" and int(x["percentage_probability"]) > 80):
                print("person" + " " + str(int(x["percentage_probability"])))
