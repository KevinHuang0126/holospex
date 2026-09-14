"""Detail-decoder behavior, pretrained reuse, and real checkpoint round trips."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ARCHITECTURE = "deeplabv3plus_mobilenet_v3_large"
CLASSES = [
    {"index": 0, "sourceId": 0, "structureId": "background"},
    {"index": 1, "sourceId": 5, "structureId": "gallbladder"},
    {"index": 2, "sourceId": 4, "structureId": "cystic_duct"},
]


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("torchvision"), "Training dependencies not installed")
class DetailModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        import torch

        torch.set_num_threads(cls.previous_threads)

    def test_main_and_aux_restore_odd_input_grid_and_use_stride_four_detail(self):
        import torch
        from holospex_ml.model import build_model

        model = build_model(3, architecture=ARCHITECTURE).eval()
        recorded = []
        hook = model.low_projection.register_forward_pre_hook(lambda _, args: recorded.append(args[0].shape))
        with torch.inference_mode():
            for height, width in ((32, 64), (37, 61)):
                result = model(torch.rand(1, 3, height, width))
                self.assertEqual(set(result), {"out", "aux"})
                for logits in result.values():
                    self.assertEqual(tuple(logits.shape), (1, 3, height, width))
                    self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(tuple(recorded[-1]), (1, 24, (height + 3) // 4, (width + 3) // 4))
        hook.remove()

    def test_singleton_training_reaches_skip_context_and_aux_with_frozen_pretrained_bn(self):
        import torch
        from holospex_ml.model import build_model, freeze_batchnorm

        model = build_model(3, architecture=ARCHITECTURE).train()
        freeze_batchnorm(model)
        result = model(torch.rand(1, 3, 32, 64))
        labels = torch.randint(0, 3, (1, 32, 64))
        loss = torch.nn.functional.cross_entropy(result["out"], labels)
        loss += 0.4 * torch.nn.functional.cross_entropy(result["aux"], labels)
        loss.backward()
        for weight in (model.low_projection[0].weight, model.low_projection[1].weight,
                       model.classifier[1].weight, model.decoder[-1].weight, model.aux_classifier[-1].weight):
            self.assertIsNotNone(weight.grad)
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                self.assertFalse(layer.training)
                self.assertTrue(all(not parameter.requires_grad for parameter in layer.parameters()))
            elif isinstance(layer, torch.nn.GroupNorm):
                self.assertTrue(layer.training)
                self.assertTrue(all(parameter.requires_grad for parameter in layer.parameters()))

    def test_generic_pretrained_backbone_and_context_are_retained_without_downloading(self):
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        from holospex_ml.model import DeepLabV3_MobileNet_V3_Large_Weights, build_model

        generic = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        backbone = generic.backbone
        aspp = generic.classifier[0]
        context = generic.classifier[1]
        auxiliary_context = generic.aux_classifier[0]
        with patch("holospex_ml.model.deeplabv3_mobilenet_v3_large", return_value=generic) as constructor:
            model = build_model(3, pretrained=True, architecture=ARCHITECTURE)
        constructor.assert_called_once_with(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
                                            weights_backbone=None, aux_loss=True)
        self.assertIs(model.backbone, backbone)
        self.assertIs(model.classifier[0], aspp)
        self.assertIs(model.classifier[1], context)
        self.assertIs(model.aux_classifier[0], auxiliary_context)
        self.assertEqual(model.aux_classifier[-1].out_channels, 3)
        self.assertEqual(model.decoder[-1].out_channels, 3)

    def test_real_tensor_only_checkpoint_round_trip_preserves_predictions(self):
        import torch
        from holospex_ml.model import build_model, load_checkpoint, model_id_for_architecture

        original = build_model(3, architecture=ARCHITECTURE).eval()
        inputs = torch.rand(1, 3, 33, 65)
        with torch.inference_mode():
            expected = original(inputs)
        checkpoint = {
            "format_version": 1,
            "architecture": ARCHITECTURE,
            "model_id": model_id_for_architecture(ARCHITECTURE),
            "classes": CLASSES,
            "input_size": {"width": 65, "height": 33},
            "model_state": original.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "detail.pt"
            torch.save(checkpoint, path)
            restored, metadata = load_checkpoint(path, "cpu")
        self.assertFalse(restored.training)
        self.assertEqual(metadata["architecture"], ARCHITECTURE)
        with torch.inference_mode():
            actual = restored(inputs)
        for head in expected:
            torch.testing.assert_close(actual[head], expected[head], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
