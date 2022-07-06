from object_detect_draw import draw
from object_detection import get_model

file_name = 'output/bear.bmp'
model = get_model()

labels, boxes, scores = model.predict_top(file_name)

draw(file_name, labels, boxes, scores)