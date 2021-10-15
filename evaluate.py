import numpy as np

def evaluation(gt_path, in_path):

    gt_label = np.load(gt_path)
    pred_label = np.load(in_path)

    print(gt_label.shape[0])
    print(pred_label.shape[0])
    evaluate_avg_precision_recall(gt_label.reshape(-1,), pred_label.reshape(-1,))

def single_evaluation(gt_path, in_path):
    gt_label = np.load(gt_path)
    pred_label = np.load(in_path)
    evaluate_single_precision_recall(gt_label.reshape(-1,),pred_label.reshape(-1,))

def evaluate_avg_precision_recall(gt_label, pred_label):
    avg_precision = []
    avg_recall = []
    for i in set(gt_label):
        if i < 0:
            continue

        pred_subset = pred_label[gt_label==i]
        pred_subset = pred_subset.astype(int)
        max_frequency = np.bincount(pred_subset).max()
        max_pred_label = np.argmax(np.bincount(pred_subset))
        recall_subset = max_frequency/pred_subset.shape[0]
        precision_subset = max_frequency/pred_label[pred_label==max_pred_label].shape[0]
        avg_recall.append(recall_subset)
        avg_precision.append(precision_subset)

    print('average precision:',np.array(avg_precision).mean())
    print('deviation precision:', np.array(avg_precision).std())
    print('average recall:', np.array(avg_recall).mean())
    print('deviation recall:', np.array(avg_recall).std())

def evaluate_single_precision_recall(gt_label, pred_label):
    for i in set(gt_label):
        if i < 0:
            continue

        pred_subset = pred_label[gt_label==i]
        pred_subset = pred_subset.astype(int)
        max_frequency = np.bincount(pred_subset).max()
        max_pred_label = np.argmax(np.bincount(pred_subset))
        recall_subset = max_frequency/pred_subset.shape[0]
        precision_subset = max_frequency/pred_label[pred_label==max_pred_label].shape[0]

        if precision_subset < 0.9:
            print('label:',i)
            print('num:',max_frequency)
            print('precision:',precision_subset)
            print('recall:',recall_subset)
            print('true label size:',pred_subset.shape)
            print('max pred label:',max_pred_label)
            print('max label size:',pred_label[pred_label==max_pred_label].shape[0])
