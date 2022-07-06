from detecto import core, utils, visualize
from torch import tensor

def draw(file_name, labels, boxes, scores):
    image = utils.read_image(file_name)
    model = core.Model()

    labels, boxes, scores = model.predict_top(image)

    boxes_list = boxes.tolist()
    scores_list = scores.tolist()

    score_box_pairs = [(score, box, label) for score, box, label in zip(scores_list, boxes_list, labels)]
    score_box_pairs.sort(reverse=True)
    top_k_filter = min(4, len(scores_list))

    scores = tensor([score_box_pairs[i][0] for i in range(top_k_filter)])
    boxes = tensor([score_box_pairs[i][1] for i in range(top_k_filter)])
    labels = [score_box_pairs[i][2] for i in range(top_k_filter)]

    visualize.show_labeled_image(image, boxes, labels)