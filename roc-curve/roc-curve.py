import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score)
    
    # Sort descending by score
    desc = np.argsort(-y_score, kind='mergesort')
    y_t = y_true[desc]
    score_d = y_score[desc]
    
    # Cumulative sums
    cum_tp = np.cumsum(y_t)
    cum_fp = np.cumsum(1 - y_t)
    
    # Find last occurrence of each unique score
    is_last = np.concatenate([np.diff(score_d) != 0, [True]])
    uniq_indices = np.where(is_last)[0]
    
    # At threshold = score[i], items 0..i are positive (score >= threshold)
    tps = cum_tp[uniq_indices]
    fps = cum_fp[uniq_indices]
    
    # Prepend (0,0) for inf threshold
    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])
    thresholds = np.concatenate([[np.inf], score_d[uniq_indices]])
    
    # Normalize
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps)
    
    return fpr.tolist(), tpr.tolist(), thresholds.tolist()