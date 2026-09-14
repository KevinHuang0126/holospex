"""Numerical IoU-surrogate and integration checks; synthetic pixels only."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HAS_TRAINING = all(importlib.util.find_spec(package) for package in ("torch", "torchvision", "PIL"))


@unittest.skipUnless(HAS_TRAINING, "Training dependencies not installed")
class LovaszTests(unittest.TestCase):
    def test_sorted_jaccard_prefixes_match_manual_foreground_objective(self):
        import torch
        from holospex_ml.losses import foreground_lovasz_softmax_loss

        # Foreground errors .8, .6, .2 sort positive, negative, positive.
        # Jaccard prefix increments are 1/2, 1/6, 1/3. Class 2 is absent.
        probabilities = torch.tensor([[[[.7, .3, .1]], [[.2, .6, .8]], [[.1, .1, .1]]]])
        target = torch.tensor([[[1, 0, 1]]])
        expected = .8 / 2 + .6 / 6 + .2 / 3
        self.assertAlmostEqual(foreground_lovasz_softmax_loss(probabilities.log(), target).item(), expected, places=6)

    def test_perfect_and_wrong_predictions_and_nonzero_gradients(self):
        import torch
        from holospex_ml.losses import foreground_lovasz_softmax_loss

        target = torch.tensor([[[0, 1, 2]]])
        correct = torch.full((1, 3, 1, 3), -40.0).scatter_(1, target[:, None], 40.0)
        self.assertAlmostEqual(foreground_lovasz_softmax_loss(correct, target).item(), 0.0, places=6)
        wrong = torch.roll(correct, shifts=1, dims=1)
        self.assertGreater(foreground_lovasz_softmax_loss(wrong, target).item(), .999)
        logits = torch.tensor([[[[.3, -.1, .7]], [[-.2, .6, .1]], [[.9, .2, -.5]]]], requires_grad=True)
        loss = foreground_lovasz_softmax_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_ignored_nonfinite_pixels_do_not_affect_loss_or_gradients(self):
        import torch
        from holospex_ml.losses import foreground_lovasz_softmax_loss

        target = torch.tensor([[[0, 1, 2]]])
        logits = torch.tensor([[[[.3, -.1, .7]], [[-.2, .6, .1]], [[.9, .2, -.5]]]], requires_grad=True)
        reference = foreground_lovasz_softmax_loss(logits, target)
        reference.backward()
        padded = torch.cat([logits.detach(), torch.full((1, 3, 1, 2), float("nan"))], dim=-1).requires_grad_()
        loss = foreground_lovasz_softmax_loss(padded, torch.cat([target, torch.full((1, 1, 2), 255)], dim=-1))
        loss.backward()
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(padded.grad[..., :3], logits.grad)
        torch.testing.assert_close(padded.grad[..., 3:], torch.zeros_like(padded.grad[..., 3:]), atol=0, rtol=0)

    def test_single_scored_pixel_keeps_pixel_axis_and_float32_accumulation(self):
        import torch
        from holospex_ml.losses import foreground_lovasz_softmax_loss

        logits = torch.tensor([[[[0., float("nan")]], [[1., float("nan")]], [[-1., float("nan")]]]], dtype=torch.float16, requires_grad=True)
        target = torch.tensor([[[1, 255]]])
        loss = foreground_lovasz_softmax_loss(logits, target)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), 1 - torch.tensor([0., 1., -1.]).softmax(0)[1].item(), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[..., 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[..., 1].abs().sum().item(), 0)

    def test_batch_pooling_includes_background_negatives_and_all_background_is_zero(self):
        import torch
        from holospex_ml.losses import foreground_lovasz_softmax_loss

        probabilities = torch.tensor([.1, .9]).view(1, 2, 1, 1).expand(2, 2, 1, 1)
        target = torch.tensor([[[1]], [[0]]])
        pooled = foreground_lovasz_softmax_loss(probabilities.log(), target)
        self.assertAlmostEqual(pooled.item(), .5, places=6)
        individual = sum(foreground_lovasz_softmax_loss(probabilities[i:i+1].log(), target[i:i+1]) for i in range(2)) / 2
        self.assertAlmostEqual(individual.item(), .05, places=6)
        logits = torch.full((1, 3, 1, 2), 1e38, requires_grad=True)
        zeros = foreground_lovasz_softmax_loss(logits, torch.zeros((1, 1, 2), dtype=torch.long))
        self.assertEqual(zeros.item(), 0)
        zeros.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=0, rtol=0)

    def test_gradient_matches_finite_differences_away_from_sort_ties(self):
        import torch
        from holospex_ml.losses import foreground_lovasz_softmax_loss

        logits = torch.tensor([[[[.3, -.1, .7]], [[-.2, .6, .1]], [[.9, .2, -.5]]]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[[0, 1, 2]]])
        self.assertTrue(torch.autograd.gradcheck(lambda value: foreground_lovasz_softmax_loss(value, target),
                                               (logits,), eps=1e-3, atol=1e-3, rtol=1e-2))

    def test_invalid_targets_shapes_and_nonfinite_scored_logits_fail(self):
        import torch
        from holospex_ml.losses import foreground_lovasz_softmax_loss

        logits = torch.zeros(1, 3, 1, 2)
        for target in (torch.full((1, 1, 2), 255), torch.tensor([[[0, -1]]]), torch.tensor([[[0, 3]]]),
                       torch.zeros(1, 1, 2), torch.zeros(1, 2, 2, dtype=torch.long)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                foreground_lovasz_softmax_loss(logits, target)
        with self.assertRaisesRegex(ValueError, "finite"):
            foreground_lovasz_softmax_loss(logits.fill_(float("nan")), torch.ones((1, 1, 2), dtype=torch.long))

    def test_training_adds_lovasz_on_main_only_and_records_actual_objective(self):
        import torch
        from PIL import Image
        from holospex_ml.losses import foreground_lovasz_softmax_loss
        from holospex_ml.training import train

        class TinyHeads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([0., 1., -1.]))

            def forward(self, image):
                values = self.logits.view(1, 3, 1, 1).expand(image.shape[0], 3, *image.shape[-2:])
                return {"out": values, "aux": values * .5}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classes = [{"index": i, "sourceId": i, "structureId": name} for i, name in enumerate(["background", "gallbladder", "cystic_duct"])]
            masks = [np.array([[0, 1, 2, 255]] * 2, dtype=np.uint8), np.array([[0, 0, 0, 1]] * 2, dtype=np.uint8),
                     np.array([[2, 2, 2, 2]] * 2, dtype=np.uint8), np.array([[0, 1, 2, 255]] * 2, dtype=np.uint8)]
            samples = []
            for i, mask in enumerate(masks):
                image_path, mask_path = root / f"{i}.jpg", root / f"{i}.png"
                Image.fromarray(np.full((2, 4, 3), 127, dtype=np.uint8)).save(image_path)
                Image.fromarray(mask).save(mask_path)
                samples.append({"split": "train" if i < 3 else "val", "videoId": i, "imagePath": str(image_path), "maskPath": str(mask_path)})
            manifest = {"classes": classes, "samples": samples, "ignoreIndex": 255, "ignoreSourceIds": [255]}
            calls = []
            def capture(logits, target):
                calls.append((logits.detach().clone(), target.detach().clone()))
                return foreground_lovasz_softmax_loss(logits, target)
            model = TinyHeads()
            with patch("holospex_ml.training.build_model", return_value=model), \
                 patch("holospex_ml.training.foreground_lovasz_softmax_loss", side_effect=capture):
                report = train(manifest, root / "run", epochs=1, batch_size=2, width=4, height=2,
                               device="cpu", pretrained=False, class_weighting="balanced", loss="ce_lovasz")
            self.assertEqual(len(calls), 2, "Only main-head train batches, including the singleton, use Lovasz")
            weights = torch.tensor(report["config"]["class_weights"])
            ce_values, lovasz_values = [], []
            for logits, target in calls:
                ce_values.append((torch.nn.functional.cross_entropy(logits, target, weight=weights, ignore_index=255)
                                  + .4 * torch.nn.functional.cross_entropy(logits * .5, target, weight=weights, ignore_index=255)).item())
                lovasz_values.append(foreground_lovasz_softmax_loss(logits, target).item())
            row, config = report["history"][0], report["config"]
            self.assertEqual(config["lovasz_weight"], .25)
            self.assertEqual(config["dice_weight"], 0)
            self.assertEqual(row["train_loss_normalizer"], 2)
            self.assertAlmostEqual(row["train_loss"], np.mean(ce_values) + .25 * np.mean(lovasz_values), places=6)
            self.assertAlmostEqual(row["train_loss_components"]["lovasz"], np.mean(lovasz_values), places=6)
            self.assertIsNone(row["train_loss_components"]["generalized_dice"])
            self.assertEqual(row["train_loss_components"]["lovasz_weight"], .25)
            self.assertEqual(config["loss_details"]["batch_objective"], "CE_main + 0.4*CE_aux + lovasz_weight*Lovasz_main")
            val_target = torch.from_numpy(masks[-1].astype(np.int64))[None]
            expected_val = torch.nn.functional.cross_entropy(model(torch.zeros(1, 3, 2, 4))["out"], val_target, ignore_index=255)
            self.assertAlmostEqual(row["validation"]["loss"], expected_val.item(), places=6)
            checkpoint = torch.load(report["checkpoint"], weights_only=True)
            self.assertEqual(checkpoint["loss"], "ce_lovasz")
            self.assertEqual(checkpoint["lovasz_weight"], .25)
            self.assertEqual(checkpoint["training"]["loss_details"], config["loss_details"])
            with patch("holospex_ml.model.build_model", return_value=TinyHeads()), \
                 patch("holospex_ml.training.build_model") as fresh_model:
                warm = train(manifest, root / "warm", epochs=1, batch_size=2, width=4, height=2,
                             device="cpu", initial_checkpoint=Path(report["checkpoint"]), loss="ce_lovasz", lovasz_weight=.4)
            fresh_model.assert_not_called()
            self.assertEqual(warm["config"]["initialization"], "checkpoint")
            self.assertEqual(warm["config"]["lovasz_weight"], .4)
            self.assertEqual(warm["history"][0]["train_loss_components"]["lovasz_weight"], .4)

    def test_cli_accepts_explicit_weight_and_invalid_weights_fail_before_model(self):
        from holospex_ml.cli import main
        from holospex_ml.training import train

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            with patch("holospex_ml.training.train", return_value={}) as runner:
                self.assertEqual(main(["train", "--manifest", str(manifest), "--output-dir", str(root / "out"),
                                       "--loss", "ce_lovasz", "--lovasz-weight", "0.4"]), 0)
            self.assertEqual(runner.call_args.kwargs["loss"], "ce_lovasz")
            self.assertEqual(runner.call_args.kwargs["lovasz_weight"], .4)
            minimal = {"classes": [{"index": 0, "sourceId": 0, "structureId": "background"},
                                   {"index": 1, "sourceId": 1, "structureId": "gallbladder"}], "samples": []}
            for value in (-1, float("nan"), float("inf"), True):
                with self.subTest(value=value), patch("holospex_ml.training.build_model") as model:
                    with self.assertRaisesRegex(ValueError, "lovasz_weight"):
                        train(minimal, root / "out", loss="ce_lovasz", lovasz_weight=value)
                    model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
