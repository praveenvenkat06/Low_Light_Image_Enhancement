import math

import cv2
import argparse
import numpy as np
import torch
import lib

if __name__ == '__main__':
    # read input image
    args = argparse.ArgumentParser().parse_args()
    image = cv2.imread(args.image)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # read class names from text file
    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # generate different colors for different classes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    # create input blob
    # ie. after subtracting means and normalizing
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0 ,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)


    # function to get the output layer names
    # in the architecture
    def get_output_layers(net):
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detection from each output layer
    # get the confidence, class id, bounding box parameters
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
        input_length = int(math.log(max_int, 2))

        net = Model()

        # loss
        mse_loss = torch.nn.MSELoss()

        # training data
        training_data = lib.data.get_training_set()

        for i in range(training_steps):

            net.zero_grad()
            for labeled_image, real_image in training_data:
                # zero the gradients on each iteration

                predicted_labels = net.forward(real_image)

                true_labels = torch.tensor(labeled_image).float()

                loss = mse_loss(predicted_labels, true_labels)
                net.backward(loss)
                net.step()
















































class Model:
    def __init__(self):
        pass
    def forward(self, file_name):
        return None, None, None

    def backward(self, loss):
        pass
    def step(self):
        pass

    def zero_grad(self):
        pass

    def predict_top(self, file):
        return [1, 2, 3]

def get_model():
    return Model()




