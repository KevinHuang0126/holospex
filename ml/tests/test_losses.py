"""Generalized Dice edge cases and training aggregation; no model downloads."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


HAS_TORCH = importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision")


@unittest.skipUnless(HAS_TORCH, "Training dependencies not installed")
class GeneralizedDiceTests(unittest.TestCase):
    def test_perfect_and_wrong_predictions_with_finite_gradients(self):
        import torch
        from holospex_ml.losses import foreground_generalized_dice_loss

        target = torch.tensor([[[0, 1, 2]]])
        perfect = torch.full((1, 3, 1, 3), -40.0).scatter_(1, target[:, None], 40.0)
        wrong = torch.roll(perfect, shifts=1, dims=1).requires_grad_()
        self.assertAlmostEqual(foreground_generalized_dice_loss(perfect, target).item(), 0.0, places=6)
        loss = foreground_generalized_dice_loss(wrong, target)
        self.assertGreater(loss.item(), 0.999)
        loss.backward()
        self.assertTrue(torch.isfinite(wrong.grad).all())

    def test_volume_weighted_foreground_ratio_matches_manual_computation(self):
        import torch
        from holospex_ml.losses import DICE_EPSILON, foreground_generalized_dice_loss

        # Foreground volumes are 1 and 3: normalized inverse-square weights 1, 1/9.
        target = torch.tensor([[[0, 1, 2, 2, 2]]])
        probabilities = torch.tensor([0.2, 0.3, 0.5]).view(1, 3, 1, 1).expand(1, 3, 1, 5)
        actual = foreground_generalized_dice_loss(probabilities.log(), target)
        expected = 1 - (2 * (0.3 + (1 / 9) * 1.5) + DICE_EPSILON) / (1 + 1.5 + (1 / 9) * (3 + 2.5) + DICE_EPSILON)
        self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_ignored_pixels_have_zero_influence_and_zero_gradient(self):
        import torch
        from holospex_ml.losses import foreground_generalized_dice_loss

        target = torch.tensor([[[0, 1, 2]]])
        logits = torch.tensor([[[[1., -1., 0.]], [[0., 2., 1.]], [[-1., 0., 2.]]]], requires_grad=True)
        reference = foreground_generalized_dice_loss(logits, target)
        reference.backward()
        padded = torch.cat([logits.detach(), torch.full((1, 3, 1, 2), float("nan"))], dim=-1).requires_grad_()
        padded_target = torch.cat([target, torch.full((1, 1, 2), 255)], dim=-1)
        loss = foreground_generalized_dice_loss(padded, padded_target)
        loss.backward()
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(padded.grad[..., :3], logits.grad)
        torch.testing.assert_close(padded.grad[..., 3:], torch.zeros_like(padded.grad[..., 3:]), atol=0, rtol=0)
        self.assertTrue(torch.isfinite(padded.grad).all())

    def test_absent_classes_are_excluded_and_all_background_retains_ce_penalty(self):
        import torch
        from holospex_ml.losses import DICE_EPSILON, foreground_generalized_dice_loss

        # Class 2 is absent. Dice directly scores only class 1, not an arbitrary
        # infinite/capped class-2 weight. Softmax still competes across all classes.
        logits = torch.tensor([0.1, 0.6, 0.3]).log().view(1, 3, 1, 1).expand(1, 3, 1, 2)
        target = torch.tensor([[[0, 1]]])
        expected = 1 - (2 * 0.6 + DICE_EPSILON) / (1 + 1.2 + DICE_EPSILON)
        self.assertAlmostEqual(foreground_generalized_dice_loss(logits, target).item(), expected, places=6)
        background = torch.zeros((1, 1, 2), dtype=torch.long)
        wrong = torch.tensor([[[[-10., -10.]], [[10., 10.]], [[-10., -10.]]]], requires_grad=True)
        dice = foreground_generalized_dice_loss(wrong, background)
        self.assertEqual(dice.item(), 0.0)
        dice.backward(retain_graph=True)
        torch.testing.assert_close(wrong.grad, torch.zeros_like(wrong), atol=0, rtol=0)
        ce = torch.nn.functional.cross_entropy(wrong, background)
        self.assertGreater(ce.item(), 10)
        ce.backward()
        self.assertGreater(wrong.grad.abs().sum().item(), 0)
        huge = torch.full((1, 3, 1, 2), 1e38, requires_grad=True)
        zero = foreground_generalized_dice_loss(huge, background)
        self.assertEqual(zero.item(), 0.0)
        zero.backward()
        torch.testing.assert_close(huge.grad, torch.zeros_like(huge), atol=0, rtol=0)

    def test_batch_pooling_is_not_mean_of_per_image_dice(self):
        import torch
        from holospex_ml.losses import foreground_generalized_dice_loss

        logits = torch.tensor([[[[0., 0.]], [[2., 2.]], [[-1., -1.]]],
                               [[[2., 2.]], [[-1., -1.]], [[0., 0.]]]])
        target = torch.tensor([[[1, 1]], [[0, 2]]])
        pooled = foreground_generalized_dice_loss(logits, target)
        per_image = sum(foreground_generalized_dice_loss(logits[i:i+1], target[i:i+1]) for i in range(2)) / 2
        self.assertGreater(abs((pooled - per_image).item()), 0.01)

    def test_invalid_targets_and_all_ignored_fail(self):
        import torch
        from holospex_ml.losses import foreground_generalized_dice_loss

        logits = torch.zeros(1, 3, 1, 2)
        for target in (torch.full((1, 1, 2), 255), torch.tensor([[[0, 3]]]), torch.tensor([[[0, -1]]]),
                       torch.zeros(1, 1, 2), torch.zeros(1, 2, 2, dtype=torch.long)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                foreground_generalized_dice_loss(logits, target)

    def test_fp16_logits_accumulate_in_float32_with_nonzero_finite_gradient(self):
        import torch
        from holospex_ml.losses import foreground_generalized_dice_loss

        logits = torch.tensor([[[[0., 0., 1.]], [[2., 1., -1.]], [[-1., 2., 0.]]]], dtype=torch.float16, requires_grad=True)
        loss = foreground_generalized_dice_loss(logits, torch.tensor([[[1, 2, 0]]]))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_tiny_training_logs_actual_head_components_and_keeps_validation_plain_ce(self):
        import torch
        from PIL import Image
        from holospex_ml.losses import foreground_generalized_dice_loss
        from holospex_ml.training import train

        classes = [{"index": i, "sourceId": i, "structureId": name} for i, name in enumerate(["background", "gallbladder", "cystic_duct"])]

        class TinyHeads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([0., 1., -1.]))

            def forward(self, image):
                values = self.logits.view(1, 3, 1, 1).expand(image.shape[0], 3, *image.shape[-2:])
                return {"out": values, "aux": values * 0.5}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []
            # Different ignored/class counts force distinct CE denominators;
            # batching leaves a singleton optimizer step at epoch end.
            masks = [np.array([[0, 1, 2, 255]] * 2, dtype=np.uint8),
                     np.array([[0, 0, 0, 1]] * 2, dtype=np.uint8),
                     np.array([[2, 2, 2, 2]] * 2, dtype=np.uint8),
                     np.array([[0, 1, 2, 255]] * 2, dtype=np.uint8)]
            for i, mask in enumerate(masks):
                image_path, mask_path = root / f"{i}.jpg", root / f"{i}.png"
                Image.fromarray(np.full((2, 4, 3), 127, dtype=np.uint8)).save(image_path)
                Image.fromarray(mask).save(mask_path)
                samples.append({"split": "train" if i < 3 else "val", "videoId": i, "imagePath": str(image_path), "maskPath": str(mask_path)})
            manifest = {"classes": classes, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255]}
            calls = []

            def capture(logits, target):
                calls.append((logits.detach().clone(), target.detach().clone()))
                return foreground_generalized_dice_loss(logits, target)

            model = TinyHeads()
            with patch("holospex_ml.training.build_model", return_value=model), \
                 patch("holospex_ml.training.foreground_generalized_dice_loss", side_effect=capture):
                report = train(manifest, root / "run", epochs=1, batch_size=2, width=4, height=2,
                               device="cpu", pretrained=False, class_weighting="balanced", loss="ce_generalized_dice", dice_weight=0.7)
            self.assertEqual(len(calls), 4)  # Main+aux on two train steps; never validation.
            weights = torch.tensor(report["config"]["class_weights"])
            steps = []
            for i in (0, 2):
                main_logits, target = calls[i]
                aux_logits = calls[i+1][0]
                ce_main = torch.nn.functional.cross_entropy(main_logits, target, weight=weights, ignore_index=255)
                ce_aux = torch.nn.functional.cross_entropy(aux_logits, target, weight=weights, ignore_index=255)
                dice_main = foreground_generalized_dice_loss(main_logits, target)
                dice_aux = foreground_generalized_dice_loss(aux_logits, target)
                scored = target[target != 255]
                steps.append({"ce": (ce_main + 0.4 * ce_aux).item(), "dice": (dice_main + 0.4 * dice_aux).item(),
                              "denominator": weights[scored].sum().item()})
            row = report["history"][0]
            expected_ce = sum(s["ce"] for s in steps) / 2
            expected_dice = sum(s["dice"] for s in steps) / 2
            expected_weighted_ce = sum(s["ce"] * s["denominator"] for s in steps) / sum(s["denominator"] for s in steps)
            self.assertEqual(row["train_batches"], 2)
            self.assertEqual(row["train_loss_normalizer"], 2)
            self.assertAlmostEqual(row["train_loss_components"]["ce"], expected_ce, places=6)
            self.assertAlmostEqual(row["train_loss_components"]["generalized_dice"], expected_dice, places=6)
            self.assertAlmostEqual(row["train_loss"], expected_ce + 0.7 * expected_dice, places=6)
            self.assertAlmostEqual(row["train_ce_denominator_weighted_mean"], expected_weighted_ce, places=6)
            self.assertGreater(abs(expected_ce - expected_weighted_ce), 0.001)
            self.assertEqual(report["config"]["loss_details"]["dice"]["epsilon"], 1e-6)
            self.assertEqual(report["config"]["training_loss_reduction"], "mean_over_optimizer_steps")
            self.assertEqual(report["config"]["selection_metric"], "foreground_macro_iou")
            val_target = torch.from_numpy(masks[-1].astype(np.int64))[None]
            expected_val = torch.nn.functional.cross_entropy(model(torch.zeros(1, 3, 2, 4))["out"], val_target, ignore_index=255)
            self.assertAlmostEqual(row["validation"]["loss"], expected_val.item(), places=6)
            checkpoint = torch.load(report["checkpoint"], weights_only=True)
            self.assertEqual(checkpoint["loss"], "ce_generalized_dice")
            self.assertEqual(checkpoint["dice_weight"], 0.7)
            self.assertEqual(checkpoint["training"]["loss_details"], report["config"]["loss_details"])


if __name__ == "__main__":
    unittest.main()
