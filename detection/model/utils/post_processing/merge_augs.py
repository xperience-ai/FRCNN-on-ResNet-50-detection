import numpy as np
import torch


def merge_aug_scores(aug_scores):
    """Merge augmented bbox scores."""
    if isinstance(aug_scores[0], torch.Tensor):
        return torch.mean(torch.stack(aug_scores), dim=0)
    else:
        return np.mean(aug_scores, axis=0)


def merge_aug_masks(aug_masks, img_metas, rcnn_test_cfg, weights=None):
    """Merge augmented mask prediction.

    Args:
        aug_masks (list[ndarray]): shape (n, #class, h, w)
        img_shapes (list[ndarray]): shape (3, ).
        rcnn_test_cfg (dict): rcnn test config.

    Returns:
        tuple: (bboxes, scores)
    """
    recovered_masks = [
        mask if not img_info[0]['flip'] else mask[..., ::-1]
        for mask, img_info in zip(aug_masks, img_metas)
    ]
    if weights is None:
        merged_masks = np.mean(recovered_masks, axis=0)
    else:
        merged_masks = np.average(
            np.array(recovered_masks), axis=0, weights=np.array(weights))
    return merged_masks
