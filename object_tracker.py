import os
import time
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import math
from multi import match

#==============

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

#================


box_count=6
dict1={}

def distance(p1 , p2): 
    x3=p1[0]
    y3=p1[1]
    x4=p2[0]
    y4=p2[1]
    return math.sqrt(math.pow(x4 - x3, 2) +
                math.pow(y4 - y3, 2) * 1.0) 

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    from tkinter.filedialog import askopenfilename
    path1=askopenfilename()
    video_path = path1
    video_path2=path1

    
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    # begin video capture
    try:
        vid2 = cv2.VideoCapture(int(video_path2))
    except:
        vid2 = cv2.VideoCapture(video_path2)
    

    out = None

    # get video ready to save locally if flag is set
  
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv2.VideoWriter('ouput.avi', codec, fps, (width, height))

    #===================
    out2 = None

    # get video ready to save locally if flag is set
  
    width2 = int(vid2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(vid2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = int(vid2.get(cv2.CAP_PROP_FPS))
    codec2 = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out2 = cv2.VideoWriter('ouput2.avi', codec2, fps2, (width2, height2))

    #===============


    c1=(0,height//2)
    c2=(width,height//2)
    c3=(width//2,0)
    c4=(width//2,height)

    

    frame_num = 0
    frame_num2 = 0
    # while video is running
    while True:

        return_value, frame = vid.read()
        if return_value:
            frame3 = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        #==========
        return_value2, frame2 = vid2.read()
        if return_value2:
            frame4=frame2
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image2 = Image.fromarray(frame2)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num2 +=1
        print('Frame2 #: ', frame_num2)
        frame_size2 = frame2.shape[:2]
        image_data2 = cv2.resize(frame2, (input_size, input_size))
        image_data2 = image_data2 / 255.
        image_data2 = image_data2[np.newaxis, ...].astype(np.float32)
        #==========


        start_time = time.time()
        
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        
        #=============
        batch_data2 = tf.constant(image_data2)
        pred_bbox2 = infer(batch_data2)
        for key, value in pred_bbox2.items():
            boxes2 = value[:, :, 0:4]
            pred_conf2 = value[:, :, 4:]

        boxes2, scores2, classes2, valid_detections2 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes2, (tf.shape(boxes2)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf2, (tf.shape(pred_conf2)[0], -1, tf.shape(pred_conf2)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        #==============

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]


        #================
        # convert data to numpy arrays and slice out unused elements
        num_objects2 = valid_detections2.numpy()[0]
        bboxes2 = boxes2.numpy()[0]
        bboxes2 = bboxes2[0:int(num_objects2)]
        scores2 = scores2.numpy()[0]
        scores2 = scores2[0:int(num_objects2)]
        classes2 = classes2.numpy()[0]
        classes2 = classes2[0:int(num_objects2)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h2, original_w2, _ = frame2.shape
        bboxes2 = utils.format_boxes(bboxes2, original_h2, original_w2)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox2 = [bboxes2, scores2, classes2, num_objects2]

        #============

        

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        print(detections[0].to_tlbr(),'&&&&&&&&')


        #==========
         # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names2 = []
        deleted_indx2 = []
        for i in range(num_objects2):
            class_indx2 = int(classes2[i])
            class_name2 = class_names[class_indx2]
            if class_name2 not in allowed_classes:
                deleted_indx2.append(i)
            else:
                names2.append(class_name2)
        names2 = np.array(names2)
        count2 = len(names2)
        # delete detections that are not in allowed_classes
        bboxes2 = np.delete(bboxes2, deleted_indx2, axis=0)
        scores2 = np.delete(scores2, deleted_indx2, axis=0)

        # encode yolo detections and feed to tracker
        features2 = encoder(frame2, bboxes2)
        detections2 = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes2, scores2, names2, features2)]

        print(detections2[0].to_tlbr(),'&&&&&&&&')

        # for i in detections2:
        #     bb=i.to_tlbr()
        #     cv2.rectangle(frame2, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,0), 2)

        # result2 = np.asarray(frame2)
        # result2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
        # out2.write(result2)

        #==========

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        cv2.line(frame, c1, c2,(255, 0, 255), 2)
        cv2.line(frame, c3, c4,(255, 0, 255), 2)
        cc1=0
        cc2=0
        cc3=0
        cc4=0
        import shutil
        try:
            shutil.rmtree('person')
        except:
            pass    
        os.mkdir('person')




        # update tracks
        mv=0
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            w1=abs(int(bbox[0])-int(bbox[2]))
            h1=abs(int(bbox[1])-int(bbox[3]))

            xc=int(bbox[0])+(w1//2)
            yc=int(bbox[1])+(h1//2)

            print(xc,yc,'========')
            print(c1,c2,c3,c4,'++++++')

            if xc>c1[0] and xc<c3[0] and yc>0 and yc<c1[1]:
                cc1+=1
            elif xc>c3[0] and xc<c2[0] and yc>0 and yc<c1[1]:
                cc2+=1
            elif xc>c1[0] and xc<c4[0] and yc>c1[1] and yc<c4[1]:
                cc3+=1
            else:
                cc4+=1    
            crop=frame3[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
            cv2.imwrite('person/%s.png'%(str(track.track_id)),crop)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            try:
                p1=dict1[track.track_id]
                p2=(xc,yc)
                d1=distance(p1,p2)
                mv+=d1
            except Exception as e:
                print(e)
                pass     
            dict1[track.track_id]=(xc,yc)    
        
        
        

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections


        if cc1<=box_count:

            cv2.putText(frame,'Count:%s'%(cc1),(10,15),0, 0.75, (255,255,255),2)
        else:
            cv2.putText(frame,'Count:%s'%(cc1),(10,15),0, 0.75, (0,0,0),2)
        
        if cc2<=box_count:
            cv2.putText(frame,'Count:%s'%(cc2),(c3[0]+10,c3[1]+15),0, 0.75, (255,255,255),2)
        else:
            cv2.putText(frame,'Count:%s'%(cc2),(c3[0]+10,c3[1]+15),0, 0.75, (0,0,0),2)
        
        if cc4<=box_count:
            cv2.putText(frame,'Count:%s'%(cc4),(c4[0]+10,c2[1]+15),0, 0.75, (255,255,255),2)
        else:
            cv2.putText(frame,'Count:%s'%(cc4),(c4[0]+10,c2[1]+15),0, 0.75, (0,0,0),2)

        if cc3<=box_count:
            cv2.putText(frame,'Count:%s'%(cc3),(c1[0]+10,c1[1]+15),0, 0.75, (255,255,255),2)
        else:
            cv2.putText(frame,'Count:%s'%(cc3),(c1[0]+10,c1[1]+15),0, 0.75, (0,0,0),2)

        cv2.putText(frame,'Total movement :%s'%(mv/len(tracker.tracks)),(10,50),0, 0.75, (0,0,255),2)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            pass
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        #========
        for i in detections2:
            bb=i.to_tlbr()
            try:
                crop2=frame4[int(bb[1]):int(bb[3]),int(bb[0]):int(bb[2])]
                id2=match(crop2)
            except Exception as e:
                print(e)  
                id2='0'  
            cv2.rectangle(frame2, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255,0,0), 2)
            cv2.rectangle(frame2, (int(bb[0]), int(bb[1]-30)), (int(bb[0])+(len('Person')+len(str(id2)))*17, int(bb[1])), (0,0,0), -1)
            cv2.putText(frame2, 'Person' + "-" + str(id2),(int(bb[0]), int(bb[1]-10)),0, 0.75, (255,255,255),2)

        result2 = np.asarray(frame2)
        result2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
        out2.write(result2)   

        #===========
       
        out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
