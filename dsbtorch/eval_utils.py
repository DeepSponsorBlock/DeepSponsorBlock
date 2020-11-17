import numpy as np
import torch


def get_start_and_end_labels(labels):
    diffs = labels[..., 1:] - labels[..., :-1]

    # 1's for frames where the start of a sponsored segment occurs
    first_start = torch.unsqueeze(labels[..., 0], dim=-1)
    start_labels = torch.cat((first_start, diffs), dim=-1) == 1
    start_labels = start_labels.long()

    # 1's for frames where the end of a sponsored segment occurs
    last_end = -torch.unsqueeze(labels[..., -1], dim=-1)
    end_labels = torch.cat((diffs, last_end), dim=-1) == -1
    end_labels = end_labels.long()

    return start_labels, end_labels


def convert_to_onehot(start_preds, end_preds):
    """
    Converts preds into a one-hot vector with 1's for sponsored frames
    """
    onehot_preds = np.zeros(len(start_preds))

    # Transform into tuples of (timestamp, is_start)
    start_preds = [(idx, True) for idx in torch.nonzero(start_preds, as_tuple=False)]
    end_preds = [(idx, False) for idx in torch.nonzero(end_preds, as_tuple=False)]
    timestamps = sorted(start_preds + end_preds, key=lambda t: t[0])

    seg_start = 0
    in_seg = False
    for t, is_start in timestamps:
        if is_start and not in_seg:
            seg_start = t
            in_seg = True
        elif not is_start and in_seg:
            onehot_preds[seg_start : t+1] = 1
            in_seg = False

    return torch.tensor(onehot_preds, dtype=torch.long)


def compute_IOU(preds, labels):
    intersection = torch.nonzero(preds * labels, as_tuple=False).shape[0]
    union = torch.nonzero(preds + labels, as_tuple=False).shape[0]
    return intersection / union if union != 0 else 0


def compute_IOU_from_indices(pred_starts, pred_ends, label_starts, label_ends):
    pred_starts, pred_ends, label_starts, label_ends = tuple(
        x.cpu().numpy() for x in
        (pred_starts, pred_ends, label_starts, label_ends))

    pred_ends += 1
    label_ends += 1
    intersection = np.maximum(0.0,
                              np.minimum(pred_ends, label_ends) - np.maximum(
                                  pred_starts, label_starts))
    union = (pred_ends - pred_starts) + (
                label_ends - label_starts) - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection),
                     where=union != 0)
