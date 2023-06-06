import numpy as np
from pred import V8
from skimage import exposure
import cv2
import glob
from sklearn.metrics import precision_recall_curve, average_precision_score
import csv
import matplotlib.pyplot as plt

csv_file=open('prec_recall.csv','w',newline='')
spamwriter=csv.writer(csv_file)
def compute_iou(box1, box2):

    # import ipdb; ipdb.set_trace()
    if len(box2)>1:
        iou=np.zeros(len(box2))
        for i in range(len(box2)):            
            x1 = max(box1[0], box2[i][0])
            y1 = max(box1[1], box2[i][1])
            x2 = min(box1[2], box2[i][2])
            y2 = min(box1[3], box2[i][3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[i][2] - box2[i][0]) * (box2[i][3] - box2[i][1])
            union = area1 + area2 - intersection
            iou[i] = intersection / union if union > 0 else 0
    
    return iou


def compute_precision_recall_at_thresholds(detections, ground_truths, iou_thresholds):
    """
    Computes precision and recall values for a list of detections and ground truth boxes
    at multiple IOU threshold levels.

    Args:
        detections: A list of detection boxes, each represented as a tuple of (x1, y1, x2, y2, score).
        ground_truths: A list of ground truth boxes, each represented as a tuple of (x1, y1, x2, y2).
        iou_thresholds: A list of IOU threshold values to evaluate.

    Returns:
        A tuple of two lists: (precisions, recalls), where precisions and recalls are lists of
        precision and recall values computed at each IOU threshold.
    """
    num_detections = len(detections)
    num_ground_truths = len(ground_truths)
    num_thresholds = len(iou_thresholds)

    # Initialize precision and recall arrays.
    precisions = np.zeros((num_thresholds,))
    recalls = np.zeros((num_thresholds,))

    # Sort the detections by decreasing confidence scores.
    # detections.sort(key=lambda x: x[4], reverse=True)

    # Initialize a list to keep track of matched ground truth boxes.

    # Loop over the detections and compute precision and recall at each IOU threshold.
    for k, threshold in enumerate(iou_thresholds):
        matched = [False] * num_ground_truths
        true_positives = 0
        false_positives = 0

        for i in range(num_detections):
            detection = detections[i]
            overlaps = compute_iou(detection, ground_truths)

            # Find the ground truth box with the highest IOU.
            max_overlap_idx = np.argmax(overlaps)
            max_overlap = overlaps[max_overlap_idx]

            if max_overlap >= threshold:
                if not matched[max_overlap_idx]:
                    true_positives += 1
                    matched[max_overlap_idx] = True
                else:
                    false_positives += 1
            else:
                false_positives += 1

        # Compute precision and recall at the current threshold.
        # import ipdb; ipdb.set_trace()
        if true_positives > 0:
            precisions[k] = true_positives / (true_positives + false_positives)
            recalls[k] = true_positives / num_ground_truths

    return precisions, recalls

def plot_precision_recall_curve(precision, recall):
    """
    Plots the precision-recall curve for object detection evaluation.

    Args:
        precision: A list of precision values.
        recall: A list of recall values.

    Returns:
        None.
    """
    # Compute the interpolated precision values at 11 recall levels.
    recall_levels = np.linspace(0, 1, 11)
    interp_precision = np.zeros_like(recall_levels)

    for i, level in enumerate(recall_levels):
        mask = recall >= level
        if np.any(mask):
            interp_precision[i] = np.max(precision[mask])
        else:
            interp_precision[i] = 0.0

    # Plot the precision-recall curve.
    fig, ax = plt.subplots()
    ax.plot(recall, precision, 'x',label='Precision-Recall curve')
    ax.plot(recall_levels, interp_precision, 'o', color='r',
            label='11-point Interpolated Precision')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend()
    plt.show()


ref='L0000.jpeg'
images_list=sorted(glob.glob('/home/pahuja/Projects/Apple tracking/Dataset/image_rect_left/*.jpeg'))[:50]
gts=sorted(glob.glob('/home/pahuja/Projects/Apple tracking/Dataset/detections2/*.csv'))[:50]
ref=cv2.imread(ref)
detector= V8(model_path='./runs/detect/train2/weights/best.pt')

def draw_boxes(image, boxes):

    for box in boxes:
        cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),(255,255,255),3)
    
    return image

# for image,gt in zip(images_list,gts): 
    # import ipdb; ipdb.set_trace()
image=images_list[0]
gt=gts[0]
image=cv2.imread(image)
# image=exposure.match_histograms(image, ref)
pred_boxes,_,pred_img=detector.pred(image.copy(),debug=True,display=False)
gt_boxes = np.loadtxt(gt,delimiter=",", dtype=int)
image=draw_boxes(image,gt_boxes)
precision,recall=compute_precision_recall_at_thresholds(gt_boxes,pred_boxes,[0.01,0.1,0.4,0.6,0.9,1])
print(precision)
print(recall)
spamwriter.writerow([str(precision),str(recall)])
plot_precision_recall_curve(precision=precision,recall=recall)

cv2.imwrite('ground truth.png',image)
cv2.imwrite('predicted.png',pred_img)

csv_file.close()
