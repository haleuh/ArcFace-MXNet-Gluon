import numpy as np
from scipy.spatial.distance import cosine


def evaluate_pairs(pair_file, features1, features2=None, kfold=10):
    if features2 is None:
        features2 = features1
    # Read pairs and labels
    pairs = np.loadtxt(pair_file, dtype=int)
    f1 = pairs[:, 0]
    f2 = pairs[:, 1]
    labels = pairs[:, 2]
    # Compare feature pairs
    sim = compare_pairs(features1[f1, :], features2[f2, :])
    # Compute the accuracy
    scores = []
    thresholds = np.arange(0, 4, 0.001)
    assert sim.shape[0] % kfold == 0
    k = int(sim.shape[0] / kfold)
    for i in range(kfold):
        _, _, acc = roc_curve(labels[i*k: (i+1)*k], sim[i*k: (i+1)*k], thresholds)
        scores.append(acc)
    scores = np.array(scores)
    scores = scores * 100
    mu_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    idx = np.argmax(mu_scores)
    return mu_scores[idx], std_scores[idx], thresholds[idx], scores[:, idx]


def compare_pairs(f1, f2):
    sim = []
    for x1, x2 in zip(f1, f2):
        sim.append(cosine(x1, x2))
    return np.array(sim)


def roc_curve(ground_truth, sim_matrix, thresholds=None):
    """Compute fpr, tpr, acc based on the input thresholds.
    If thresholds is None, use thresholds arrange from  (min, max, 0.001)
    of similarity matrix. sim_matrix lower is better
    Args:
        ground_truth: 0 for negative, 1 for positive
        sim_matrix: lower than threshold is positive
        thresholds: thresholds used to compare

    Returns:
        fps, tps, accuracies: the same length as thresholds
    """
    if thresholds is None:
        thresholds = np.arange(sim_matrix.min(), sim_matrix.max(), 0.001)
    n = len(thresholds)
    fps = np.zeros(n, dtype=np.float32)
    tps = np.zeros(n, dtype=np.float32)
    accuracies = np.zeros(n, dtype=np.float32)
    for i, threshold in enumerate(thresholds):
        predict = np.less_equal(sim_matrix, threshold)
        tp = np.sum(np.logical_and(predict, ground_truth))
        fp = np.sum(np.logical_and(predict, np.logical_not(ground_truth)))
        tn = np.sum(np.logical_and(np.logical_not(predict), np.logical_not(ground_truth)))
        fn = np.sum(np.logical_and(np.logical_not(predict), ground_truth))

        tpr = 0 if (tp + fn == 0) else tp / (tp + fn)
        fpr = 0 if (fp + tn == 0) else fp / (fp + tn)
        acc = (tp + tn) / sim_matrix.size
        fps[i] = fpr
        tps[i] = tpr
        accuracies[i] = acc
    return fps, tps, accuracies
