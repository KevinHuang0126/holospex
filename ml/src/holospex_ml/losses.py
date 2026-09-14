"""Explicit foreground overlap losses for Holospex training.

Sudre et al. (2017), https://arxiv.org/abs/1707.03237, motivates inverse-squared
target-volume weighting. Foreground-only batch pooling, absent-class exclusion,
weight rescaling, smoothing, and combination with CE are our choices; this is
not an exact reproduction of the paper's experiments.

The Lovasz implementation below adapts the sorted Jaccard-gradient algorithm
from Berman, Rannen Triki, and Blaschko (CVPR 2018),
https://arxiv.org/abs/1705.08790 and their official MIT-licensed implementation:
https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py

Copyright (c) 2018 Maxim Berman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .metrics import IGNORE_INDEX

DICE_EPSILON = 1e-6


def generalized_dice_policy() -> dict:
    """JSON-safe exact policy, stored with every enabled training objective."""
    return {
        "reference": "https://arxiv.org/abs/1707.03237",
        "variant": "foreground_present_classes_batch_pooled_generalized_soft_dice",
        "probabilities": "float32_softmax_over_all_model_channels",
        "scope": "all_scored_pixels_across_batch_and_spatial_dimensions",
        "classes": "foreground_training_indices_1_through_C_minus_1",
        "class_weights": "(minimum_present_foreground_target_volume / class_target_volume)^2",
        "weight_rescaling": "common_scale_equivalent_to_inverse_squared_target_volumes_before_smoothing; max_present_weight_1",
        "target_volumes": "scored_target_pixels_in_current_batch; detached_from_autograd",
        "absent_class_policy": "zero_Dice_weight_when_target_volume_is_zero; CE_still_penalizes_false_positives",
        "background_policy": "excluded_from_Dice_sums; included_in_softmax_and_CE",
        "all_background_policy": "differentiable_zero_Dice_loss; CE_remains_active",
        "all_ignored_policy": "error_no_supervision",
        "ignore_index": IGNORE_INDEX,
        "ignore_policy": "mask_logits_before_softmax_and_exclude_targets_predictions_and_intersections",
        "epsilon": DICE_EPSILON,
        "formula": "1 - (2 * sum_c(w_c * intersection_c) + epsilon) / (sum_c(w_c * (target_volume_c + predicted_volume_c)) + epsilon)",
        "paper_deviations": ["foreground_only", "batch_pooling", "absent_class_exclusion", "common_weight_rescaling_and_epsilon", "additive_cross_entropy"],
    }


def foreground_generalized_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return one nonadditive Dice loss per batch, with no ignored-pixel gradient.

    Present foreground classes retain the inverse-squared-volume weight ratio.
    Rescaling their maximum weight to one avoids tiny sums at large image sizes.
    An absent class has zero direct Dice weight; retained all-class CE supplies
    its explicit false-positive penalty, including in all-background batches.
    """
    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:] or logits.shape[1] < 2:
        raise ValueError("Dice requires N,C,H,W logits and aligned N,H,W targets, with C >= 2.")
    if not logits.is_floating_point() or target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Dice requires floating logits and integer class targets.")
    if target.device != logits.device:
        raise ValueError("Dice logits and targets must use the same device.")
    scored = target != IGNORE_INDEX
    if not bool(scored.any()):
        raise ValueError("Dice batch contains only ignored pixels; no supervision remains.")
    if bool(((target < 0) | (target >= logits.shape[1]))[scored].any()):
        raise ValueError("Dice targets must be model class indices or ignored index 255.")

    # Mask before softmax so even a nonfinite ignored logit cannot contaminate
    # the loss. masked_fill also gives every ignored logit an exact zero gradient.
    safe_logits = logits.float().masked_fill(~scored[:, None], 0.0)
    probabilities = safe_logits.softmax(dim=1) * scored[:, None]
    safe_target = target.masked_fill(~scored, 0).long()
    truth = F.one_hot(safe_target, num_classes=logits.shape[1]).movedim(-1, 1).to(torch.float32)
    truth = truth * scored[:, None]
    truth, probabilities = truth[:, 1:], probabilities[:, 1:]
    axes = (0, 2, 3)
    volumes = truth.sum(dim=axes)
    present = volumes > 0
    if not bool(present.any()):
        # Probabilities are bounded; a sum of huge finite raw logits can overflow
        # before multiplication by zero even when softmax is perfectly finite.
        return probabilities.sum() * 0.0
    present_volumes = volumes[present]
    weights = (present_volumes.min() / present_volumes).square()
    intersections = (truth * probabilities).sum(dim=axes)[present]
    predicted_volumes = probabilities.sum(dim=axes)[present]
    numerator = 2.0 * (weights * intersections).sum() + DICE_EPSILON
    denominator = (weights * (present_volumes + predicted_volumes)).sum() + DICE_EPSILON
    return 1.0 - numerator / denominator


def lovasz_policy() -> dict:
    """Exact, JSON-safe policy for the optional main-head IoU surrogate."""
    return {
        "reference": "https://arxiv.org/abs/1705.08790",
        "implementation_reference": "https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py",
        "variant": "foreground_present_classes_batch_pooled_lovasz_softmax",
        "probabilities": "float32_softmax_over_all_model_channels_after_selecting_scored_pixels",
        "scope": "all_scored_pixels_across_batch_and_spatial_dimensions; main_head_only",
        "classes": "foreground_training_indices_1_through_C_minus_1_present_in_scored_targets",
        "class_reduction": "arithmetic_mean_over_present_foreground_classes",
        "absent_class_policy": "excluded_from_Lovasz_mean; CE_still_penalizes_false_positives",
        "background_policy": "excluded_from_Lovasz_class_mean; included_as_negative_pixels_and_in_softmax_and_CE",
        "all_background_policy": "differentiable_zero_Lovasz_loss; CE_remains_active",
        "all_ignored_policy": "error_no_supervision",
        "ignore_index": IGNORE_INDEX,
        "ignore_policy": "select_scored_pixels_before_softmax_flattening_errors_and_sorting; zero_ignored_logit_gradient",
        "sorting": "descending_absolute_probability_error_per_class; stable_ties",
        "gradient": "successive_differences_of_Jaccard_loss_for_sorted_error_prefixes",
        "formula": "mean_present_foreground_c(dot(sorted(abs(1[target=c]-p_c)), discrete_Jaccard_prefix_gradient_c))",
        "objective_adaptations": ["foreground_only", "batch_pooling", "main_head_only", "additive_cross_entropy"],
    }


def foreground_lovasz_softmax_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lovasz-Softmax on present foreground classes, pooled across the batch.

    All scored pixels, including background negatives, affect each present
    foreground class's sorted errors. Ignored pixels never enter the sorting
    problem. Boolean selection retains a pixel axis even for one valid pixel.
    """
    if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:] or logits.shape[1] < 2:
        raise ValueError("Lovasz requires N,C,H,W logits and aligned N,H,W targets, with C >= 2.")
    if not logits.is_floating_point() or target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("Lovasz requires floating logits and integer class targets.")
    if target.device != logits.device:
        raise ValueError("Lovasz logits and targets must use the same device.")
    scored = target != IGNORE_INDEX
    if not bool(scored.any()):
        raise ValueError("Lovasz batch contains only ignored pixels; no supervision remains.")
    labels = target[scored]
    if bool(((labels < 0) | (labels >= logits.shape[1])).any()):
        raise ValueError("Lovasz targets must be model class indices or ignored index 255.")
    # Ignore before any softmax, so NaNs in an ignored region have no forward
    # influence and receive exactly zero gradient from this objective.
    selected_logits = logits.movedim(1, -1)[scored].float()
    if not bool(torch.isfinite(selected_logits).all()):
        raise ValueError("Lovasz scored logits must be finite in float32.")
    probabilities = selected_logits.softmax(dim=1)
    losses = []
    for class_index in range(1, logits.shape[1]):
        positive = labels == class_index
        if not bool(positive.any()):
            continue
        indicator = positive.to(torch.float32)
        errors, order = (indicator - probabilities[:, class_index]).abs().sort(descending=True, stable=True)
        sorted_indicator = indicator[order]
        positive_count = indicator.sum()
        remaining_intersection = positive_count - sorted_indicator.cumsum(dim=0)
        expanded_union = positive_count + (1.0 - sorted_indicator).cumsum(dim=0)
        prefix_jaccard_loss = 1.0 - remaining_intersection / expanded_union
        coefficients = torch.diff(prefix_jaccard_loss, prepend=prefix_jaccard_loss.new_zeros(1))
        losses.append(torch.dot(errors, coefficients))
    if not losses:
        return probabilities.sum() * 0.0
    return torch.stack(losses).mean()
