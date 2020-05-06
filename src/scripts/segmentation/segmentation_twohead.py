from pathlib import Path, PurePath
from types import SimpleNamespace

from inc.config_snake.config import ConfigFile, ConfigDict
from src.utils.segmentation.data import build_dataloaders
from src.utils.segmentation.general import set_segmentation_input_channels
from src.scripts.segmentation.preprocess import *

from src.scripts.segmentation.model import Model

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
    for k, v in config.__dict__.items():
        config.__dict__[k] = ConfigDict(config, v)

    assert config.training.validation_mode == "IID"
    assert "TwoHead" in config.architecture.name
    assert config.architecture.head_A_num_classes >= config.architecture.num_classes
    assert config.architecture.head_B_num_classes == config.architecture.num_classes

    if "last_epoch" not in config.__dict__:
        config.last_epoch = 0

    config.out_dir = str(Path(config.output.root).resolve() / str(config.dataset.id))
    out_dir = Path(config.out_dir).resolve()
    if not (out_dir.is_dir() and out_dir.exists()):
        out_dir.mkdir(parents=True, exist_ok=True)

    config.dataloader_batch_size = int(
        config.training.batch_size / config.training.num_dataloaders
    )
    config.output_k = config.architecture.head_B_num_classes  # for eval code
    config.eval_mode = "hung"
    # TODO better mechanism for identifying number of channels in data
    # TODO more robust transform into desired number of channels
    set_segmentation_input_channels(config)

    train(config)


def train(config):
    preparer = Preparer(config)
    transformation = Transformation(config)
    preprocessor = Preprocessor(config, preparer, transformation)
    head_A, map_assign, map_test = build_dataloaders(config, preprocessor)
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
    out = PurePath("out")
    shutil.rmtree(out, ignore_errors=True)
    interface()
