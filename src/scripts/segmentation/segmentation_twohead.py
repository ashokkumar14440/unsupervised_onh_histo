from pathlib import Path, PurePath
from types import SimpleNamespace

from inc.config_snake.config import ConfigFile, ConfigDict
from src.utils.segmentation.data import build_dataloaders
from src.utils.segmentation.general import set_segmentation_input_channels

from model import Model

"""
  Fully unsupervised clustering for segmentation ("IIC" = "IID").
  Train and test script.
  Network has two heads, for overclustering and final clustering.
"""


def interface():
    config_file = ConfigFile("config.json")
    config = config_file.segmentation
    config = config.to_json()
    config = SimpleNamespace(**config)
    config.dataset = ConfigDict(config, config.dataset)
    config.preprocessor = ConfigDict(config, config.preprocessor)
    config.transformations = ConfigDict(config, config.transformations)

    assert config.mode == "IID"
    assert "TwoHead" in config.arch
    assert config.output_k_A >= config.gt_k
    assert config.output_k_B == config.gt_k

    if "last_epoch" not in config.__dict__:
        config.last_epoch = 0
    config.out_dir = str(Path(config.out_root).resolve() / str(config.model_ind))
    config.dataloader_batch_size = int(
        config.dataset.batch_size / config.dataset.num_dataloaders
    )
    config.output_k = config.output_k_B  # for eval code
    config.eval_mode = "hung"
    # TODO better mechanism for identifying number of channels in data
    # TODO more robust transform into desired number of channels
    set_segmentation_input_channels(config)

    out_dir = Path(config.out_dir).resolve()
    if not (out_dir.is_dir() and out_dir.exists()):
        out_dir.mkdir(parents=True, exist_ok=True)

    train(config)


def train(config):
    head_A, map_assign, map_test = build_dataloaders(config)
    dataloaders = {
        "A": head_A,
        "B": head_A,
        "map_assign": map_assign,
        "map_test": map_test,
    }
    model = Model(config, dataloaders)
    model.train()


import shutil

if __name__ == "__main__":
    out = PurePath("out/555")
    shutil.rmtree(out, ignore_errors=True)
    interface()
