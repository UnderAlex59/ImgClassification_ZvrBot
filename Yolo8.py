import numpy as np
import os
import cv2
from ultralytics import YOLO

def classify_yolo8(img_path):
    tm = cv2.TickMeter()
    tm.reset()
    # define some parameters
    CONFIDENCE = 0.5
    font_scale = 1
    thickness = 1

    # loading the YOLOv8 model with the default weight file
    model = YOLO("yolov8x.pt")

    # loading all the class labels (objects)
    labels = open("data/coco.names").read().strip().split("\n")

    # generating colors for each object for later plotting
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    #read image
    image = cv2.imread(img_path)
    image = cv2.copyMakeBorder(image, top = 50, bottom = 50, left = 50, right = 50, borderType= cv2.BORDER_CONSTANT, value= 0)
    file_name = os.path.basename(img_path)  # "dog.jpg"
    filename, ext = file_name.split(".")  # "dog", "jpg"

    tm.start()
    results = model.predict(image, conf=CONFIDENCE)[0]
    tm.stop()
    time = tm.getTimeMilli()


    for data in results.boxes.data.tolist():
        # get the bounding box coordinates, confidence, and class id
        xmin, ymin, xmax, ymax, confidence, class_id = data
        # converting the coordinates and the class id to integers
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        class_id = int(class_id)

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_id]]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
        text = f"{labels[class_id]}: {confidence:.2f}"
        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = \
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = xmin
        text_offset_y = ymin - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        # add opacity (transparency to the box)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        # now put the text (label: confidence %)
        cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    cv2.imwrite(filename + "_yolo8.jpg" , image)
    return format("Время классификации: {:.2f} мс".format(time))
