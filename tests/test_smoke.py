"""
Minimal smoke test, no framework. Run with:  python tests/test_smoke.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from net.model.gravitynet.GravityNet import GravityNet
from net.parameters.parameters import parameters_parsing
from net.parameters.parameters_choices import parameters_choices


def test_yaml_precedence(tmp_dir: Path):
    cfg_path = tmp_dir / "test.yaml"
    cfg_path.write_text("epochs: 50\nlearning_rate: 0.001\n")

    sys.argv = ["GravityNet.py", "train", "--cfg", str(cfg_path), "--epochs", "5"]
    parser = parameters_parsing()

    assert parser.epochs == 5, "CLI must override YAML"
    assert parser.learning_rate == 0.001, "YAML must override the built-in default"
    assert parser.batch_size_train == 8, "unset params must fall back to the built-in default"

    print("test_yaml_precedence: OK")


def test_all_backbones_parse():
    for backbone in parameters_choices['backbone']:
        sys.argv = ["GravityNet.py", "train", "--backbone", backbone]
        parser = parameters_parsing()
        assert parser.backbone == backbone

    print(f"test_all_backbones_parse: OK ({len(parameters_choices['backbone'])} backbones)")


def test_forward_pass_cpu():
    net = GravityNet(backbone="ResNet-18", pretrained=False, num_gravity_points_feature_map=9)
    net.eval()

    image = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        classification, regression = net(image)

    assert classification.shape[0] == 1 and classification.shape[2] == 2
    assert regression.shape[0] == 1 and regression.shape[2] == 2
    assert classification.shape[1] == regression.shape[1]

    print(f"test_forward_pass_cpu: OK classification={tuple(classification.shape)} regression={tuple(regression.shape)}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        test_yaml_precedence(Path(tmp))

    test_all_backbones_parse()
    test_forward_pass_cpu()

    print("\nAll smoke tests passed.")
