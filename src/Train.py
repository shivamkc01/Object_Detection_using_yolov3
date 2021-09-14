import cv2
import numpy as np

camera = cv2.VideoCapture(0)
widthHeight = 320
classFile = 'coco.names'
classNames = []
confThreshold = 0.5
nmsThreshold = 0.2  # the lower it is the more aggressive it will make less no of boxes
# were are opening coco.names file and extract all the values
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
print(len(classNames))

modelConfiguration = 'yolov3-320.cfg'
modelWeights = 'yolov3-320.weights'

# creating our networks
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findObjects(outputs, img):
    height, width, channel = img.shape
    Bounding_box = []  # contains -> x, y, w, h
    classIds = []  # contains all class IDS
    confs = []  # confidence value

    for output in outputs:
        # for one array values we use detection
        for detection in output:
            scores = detection[5:]  # leaving first 5 columns, starting from class ids
            classId = np.argmax(scores)  # getting  highest confidence scores class ids using np.argmax
            confidence = scores[classId]
            if confidence > confThreshold:
                # we are going to save width & height
                # width is in element number 2
                # height is in element number 3
                # check in image in image folder

                # Here (percentage) values are in decimal so we need to multiply
                # with height, width
                # we will get pixels value not the percentage values
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int((detection[0] * width) - w / 2), int((detection[1] * height) - h / 2)  # Here x & y are the
                # center point  (x*width - w/2)
                Bounding_box.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # For taking care of overlapping boxes
    indices = cv2.dnn.NMSBoxes(Bounding_box, confs, confThreshold, nmsThreshold)
    # print(indices)
    for i in indices:
        i = i[0]
        box = Bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # cv2.rectangle(img, (x, y), conner points, color, thickness)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()}{int(confs[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # classIds gives us the index of idx,
        # y-10 because it won't touch the boundary box it shift up
        # 0.6  is for scale
        # Color (255, 0, 255)
        # Thickness = 2


while True:
    success, img = camera.read()

    # convert the img to blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (widthHeight, widthHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    # print(layersNames) --> getting our yolo layers Name
    # print(net.getUnconnectedOutLayers()) --> Here are getting the index no names
    #  We need to minus 1 in the layers index like [[200] -> [199]
    #  [227] -> [226]
    #  [254]]-> [253]
    outputNames = [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames) --> Now we getting our output layers names

    outputs = net.forward(outputNames)
    # print(type(outputs)) --> list class
    # print(outputs[0].shape) --> (300, 85)  300->rows, 85->columns
    # print(outputs[1].shape) --> (1200, 85)  1200->rows, 85->columns
    # print(outputs[2].shape) --> (4800, 85)  4800->rows, 85->columns
    """
    for creating bounding Box we only need are
    1. center cx axis
    2. center cy axis
    3. Height
    4. Width
    5. center position
    """
    # Go to the image folder and see image you will understand what is (300 X 85)
    # print(outputs[0][0])
    findObjects(outputs, img)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
