"""Exercise isolated training and its pre-write validation with a small stand-in model.

Owner: A (Data/ML).
"""

import json

import numpy as np
import pytest
import torch

from model import train


class Tiny(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = torch.nn.Linear(config["n_features"], len(config["classes"]))

    def forward(self, features):
        return self.head(features.mean(dim=1))


def alternate_data(tmp_path, n_classes):
    folder = tmp_path / "alternate_data"
    folder.mkdir()
    vocab = [{"id": f"new_class_{i}", "gloss": f"NEW {i}", "az": f"new {i}",
              "dataset_label": f"NEW {i}"} for i in range(n_classes)]
    (folder / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    return folder, vocab


def fake_splits(monkeypatch, train_labels, val_labels):
    calls = []

    def load_split(split, camera, **kwargs):
        calls.append((split, camera, kwargs))
        labels = np.asarray(train_labels if split == "train" else val_labels, dtype=np.int64)
        features = np.zeros((len(labels), train.T, train.N_FEATURES), dtype=np.float32)
        report = {"cached": True, "missing": {}, "cut": 0, "fallback": 0}
        return features, labels, report

    monkeypatch.setattr(train, "load_split", load_split)
    monkeypatch.setattr(train, "build_model", Tiny)
    return calls


def test_training_uses_alternate_vocabulary_and_isolated_cache(tmp_path, monkeypatch):
    data_dir, vocab = alternate_data(tmp_path, 60)
    output = tmp_path / "new_artifacts"
    calls = fake_splits(monkeypatch, np.arange(60), np.arange(60))
    train.main(["--data-dir", str(data_dir), "--out", str(output), "--camera", "any",
                "--epochs", "1", "--batch", "60", "--rebuild-cache"])

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["classes"] == [entry["id"] for entry in vocab]
    weights = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    assert weights["head.weight"].shape == (60, train.N_FEATURES)
    expected_cache = data_dir / "landmarks" / f"features_v{train.FEATURE_VERSION}.npz"
    assert [call[:2] for call in calls] == [("train", "any"), ("val", "any")]
    assert all(call[2]["data_dir"] == data_dir and call[2]["cache"] == expected_cache for call in calls)
    assert calls[0][2]["rebuild"] is True and "rebuild" not in calls[1][2]
    assert expected_cache != train.ROOT / "data" / "landmarks" / f"features_v{train.FEATURE_VERSION}.npz"


@pytest.mark.parametrize("n_classes, train_labels, val_labels", [
    (0, [], []),
    (2, [], [0, 1]),
    (2, [0, 1], []),
    (2, [0, 0], [0, 1]),
])
def test_incomplete_training_data_never_writes_artifacts(tmp_path, monkeypatch,
                                                       n_classes, train_labels, val_labels):
    data_dir, _ = alternate_data(tmp_path, n_classes)
    output = tmp_path / "new_artifacts"
    fake_splits(monkeypatch, train_labels, val_labels)
    with pytest.raises(SystemExit, match="Nothing written"):
        train.main(["--data-dir", str(data_dir), "--out", str(output), "--camera", "any", "--epochs", "1"])
    assert not output.exists()


class EncoderTiny(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = torch.nn.Linear(config["n_features"], 4)
        self.head = torch.nn.Linear(4, len(config["classes"]))


def checkpoint_config(classes, arch="gru"):
    return {"arch": arch, "feature_version": train.FEATURE_VERSION, "n_features": train.N_FEATURES,
            "T": train.T, "classes": classes, "temperature": 1.0, "tau": 0.7, "margin": 0.15}


def save_checkpoint(directory, config, state):
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.save(state, directory / "model.pt")


@pytest.mark.parametrize("arch", ["gru", "transformer"])
def test_initialize_reuses_encoder_and_maps_head_rows_by_class_id(tmp_path, arch):
    old_config = checkpoint_config(["a", "b", "retired"], arch)
    old_config.update(temperature=0.3, tau=0.95, margin=0.2)
    old_model = EncoderTiny(old_config)
    with torch.no_grad():
        old_model.encoder.weight.fill_(2)
        old_model.encoder.bias.fill_(3)
        for row in range(3):
            old_model.head.weight[row].fill_(10 + row)
            old_model.head.bias[row].fill_(20 + row)
    directory = tmp_path / "old_checkpoint"
    save_checkpoint(directory, old_config, old_model.state_dict())
    config = checkpoint_config(["b", "new", "a"], arch)
    new_model = EncoderTiny(config)
    new_row, new_bias = new_model.head.weight[1].detach().clone(), new_model.head.bias[1].detach().clone()

    assert train.initialize_from(new_model, config, directory) == 2
    assert torch.equal(new_model.encoder.weight, old_model.encoder.weight)
    assert torch.equal(new_model.encoder.bias, old_model.encoder.bias)
    assert torch.equal(new_model.head.weight[0], old_model.head.weight[1])
    assert torch.equal(new_model.head.bias[0], old_model.head.bias[1])
    assert torch.equal(new_model.head.weight[2], old_model.head.weight[0])
    assert torch.equal(new_model.head.bias[2], old_model.head.bias[0])
    assert torch.equal(new_model.head.weight[1], new_row)
    assert torch.equal(new_model.head.bias[1], new_bias)
    assert (config["temperature"], config["tau"], config["margin"]) == (1.0, 0.7, 0.15)


@pytest.mark.parametrize("changes", [
    {"arch": "transformer"}, {"feature_version": 999}, {"n_features": 191}, {"T": 16},
    {"classes": ["a", "a"]}, {"classes": []}, {"classes": ["Invalid ID"]},
])
def test_incompatible_checkpoint_leaves_model_untouched(tmp_path, changes):
    config = checkpoint_config(["a", "b"])
    model = EncoderTiny(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    previous = {**config, **changes}
    directory = tmp_path / "old_checkpoint"
    save_checkpoint(directory, previous, before)
    with pytest.raises(ValueError):
        train.initialize_from(model, config, directory)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_corrupt_encoder_shape_is_refused_before_changing_model(tmp_path):
    config = checkpoint_config(["a", "b"])
    model = EncoderTiny(config)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    corrupt = {**before, "encoder.weight": torch.zeros(3, train.N_FEATURES)}
    directory = tmp_path / "old_checkpoint"
    save_checkpoint(directory, config, corrupt)
    with pytest.raises(ValueError, match="encoder.weight"):
        train.initialize_from(model, config, directory)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("classes", [["a", "a"], [], ["Invalid ID"], ["a", 1], "a"])
def test_invalid_new_class_ids_are_refused(tmp_path, classes):
    config = checkpoint_config(["a", "b"])
    model = EncoderTiny(config)
    directory = tmp_path / "old_checkpoint"
    save_checkpoint(directory, config, model.state_dict())
    with pytest.raises(ValueError, match="new config needs"):
        train.initialize_from(model, {**config, "classes": classes}, directory)


def test_incompatible_initialization_never_writes_output_artifacts(tmp_path, monkeypatch):
    data_dir, _ = alternate_data(tmp_path, 2)
    fake_splits(monkeypatch, [0, 1], [0, 1])
    previous = checkpoint_config(["new_class_0", "new_class_1"], arch="transformer")
    directory = tmp_path / "old_checkpoint"
    save_checkpoint(directory, previous, Tiny(previous).state_dict())
    output = tmp_path / "new_artifacts"
    with pytest.raises(SystemExit, match="incompatible arch.*Nothing written"):
        train.main(["--data-dir", str(data_dir), "--out", str(output), "--camera", "any",
                    "--init-from", str(directory), "--epochs", "1"])
    assert not output.exists()
