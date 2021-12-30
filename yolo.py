import numpy as np
import argparse
import imutils
import time
import cv2

import os
import PIL
from tkinter import *
from timeit import default_timer as timer
import math 

font = cv2.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
red = (0, 0, 255)
line_type = cv2.LINE_AA
IMAGE_SIZE = 224
MHI_DURATION = 1500 # milliseconds
THRESHOLD = 32
GAUSSIAN_KERNEL = (3, 3)

  
def distance(x1 , y1 ,w1,h1, x2 , y2,w2,h2): 
    x3=(x1+w1)/2
    y3=(y1+h1)/2
    x4=(x2+w2)/2
    y4=(y2+h2)/2



  
    return math.sqrt(math.pow(x4 - x3, 2) +
                math.pow(y4 - y3, 2) * 1.0) 

abx,aby,abh,abw=0,0,0,0
a123=0

def work(path):
        import os
 
        dir = 'violations'
        for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        print('entered')
        vs = cv2.VideoCapture(path)
        print('yes')
        if not vs.isOpened():
                print("Cannot open video/webcam {}".format(path))
                return

        
        

    
        
        global abx,aby,abh,abw,a123
        labelsPath = os.path.sep.join(["allmodel", "labels.names"])
        LABELS = open(labelsPath).read().strip().split("\n")
        detections=["person"]
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                dtype="uint8")

        weightsPath = os.path.sep.join(["allmodel", "yolov4.weights"])
        configPath = os.path.sep.join(["allmodel", "yolov4.cfg"])

        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        
        
                   
        print("camera start")
        writer = 0
        (W, H) = (None, None)
        aa=0
        di=0.0
        v1,v2,v3,v4=0,0,0,0
        cnt=0
        vcnt=0
        while True:
                
                (grabbed, frame) = vs.read()
                frame1=frame
                

                

                

                

                if not grabbed:
                        break

                if W is None or H is None:
                        (H, W) = frame1.shape[:2]
                
                blob = cv2.dnn.blobFromImage(frame1, 1 / 255.0, (416, 416),
                        swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()
                boxes = []
                confidences = []
                classIDs = []
                # frame1=cv2.resize(frame,(500,500))
                
                

                for output in layerOutputs:
                        for detection in output:
                                scores = detection[5:]
                                classID = np.argmax(scores)
                                confidence = scores[classID]
                                
                                if confidence > 0.2:
                                        
                                        box = detection[0:4] * np.array([W, H, W, H])
                                        
                                        (centerX, centerY, width, height) = box.astype("int")

                                        x = int(centerX - (width / 2))
                                        y = int(centerY - (height / 2))

                                        boxes.append([x, y, int(width), int(height)])
                                        confidences.append(float(confidence))
                                        classIDs.append(classID)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.2,0.2)
                a123+=1
                ccc=1

                checked=[]
                violations=[]
                people=[]
                flag=0
                if len(idxs) > 0:
                        for i in idxs.flatten():
                            if LABELS[classIDs[i]] in detections:
                                    color = [int(c) for c in COLORS[classIDs[i]]]
                                    if LABELS[classIDs[i]] in ['person']:
                                            (x, y) = (boxes[i][0], boxes[i][1])
                                            (w, h) = (boxes[i][2], boxes[i][3])
                                            if h<100 or y<0 or x<0:
                                                     continue
                                       
                                            ccc=0

                                            people.append((x,y,w,h))

                                            for j in idxs.flatten():
                                                   if (i,j) in checked or (j,i) in checked:
                                                        #    print('checked')
                                                           continue
                                                   checked.append((i,j))
                                                   checked.append(((j,i)))
                                                   if j==i:
                                                        #    print('me')
                                                           continue 
                                                
                                                   if LABELS[classIDs[j]]=='person':
                                                           (x1, y1) = (boxes[j][0], boxes[j][1])
                                                           (w1, h1) = (boxes[j][2], boxes[j][3])
                                                           if h1<100 or y1<0 or x1<0:
                                                                   continue

                                                           if abs(h1-h)<30 and h1>100 and h>100:

                                                                di="%.3f"%distance(x, y,w,h, x1, y1,w1,h1)
                                                                di=float(di)
                                                                
                                                                
                                                                if float(di)<50.0:
                                                                        print(di,'distance')
                                                                        if (x,y,w,h) not in violations:
                                                                                violations.append((x,y,w,h))
                                                                        if (x1,y1,w1,h1) not in violations:
                                                                                violations.append((x1,y1,w1,h1))        
                                                                        

                                                                        flag=1
                                                                
                                            print(cnt)

                                            if flag:
                                                   text = "{}".format("Social distancing violation detected")
                                                   cv2.putText(frame1, text, (20, 20),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 2) 

                                           
                                                   
                                                          
                                            else:
                                                   pass
                                                #    cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
                                                #    text = "{}".format(LABELS[classIDs[i]])
                                                #    cv2.putText(frame1, text, (x, y - 5),
                                                #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                                #    data= str(LABELS[classIDs[i]])
                                    else:
                                        pass
                                       
#                 
#                 top.update()
                # print(violations)
                for i in violations:
                        vcnt+=1
                        print((i[0], i[1]), (i[0]+ i[2], i[1] + i[3]))
                        cv2.rectangle(frame1, (i[0], i[1]), (i[0]+ i[2], i[1] + i[3]), (0,0,255), 2)
                        crop_img = frame[i[1]:i[1] + i[3], i[0]:i[0]+ i[2]]
                        cv2.imwrite('violations/img%s.png'%(vcnt),crop_img)
                       
                for i in people:
                        if i not in violations:
                                cv2.rectangle(frame1, (i[0], i[1]), (i[0]+ i[2], i[1] + i[3]), (0,255,0), 2)


                frame=cv2.resize(frame1,(800,600))                   
                # photo1 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                # cn1.create_image(0, 0, image = photo1, anchor = NW)
                # top.update()
                cv2.imshow('win',frame)
                key = cv2.waitKey(10)
                if key == 27:
                    print("STOPED")
                    break
                # key = cv2.waitKey(10)
                
        print("[INFO] cleaning up...")
        vs.release()

work('test.mp4')

    
 

##showface()
