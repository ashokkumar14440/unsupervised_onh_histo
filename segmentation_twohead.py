from pathlib import Path, PurePath
from types import SimpleNamespace

from inc.config_snake.config import ConfigFile, ConfigDict
from src.utils.segmentation.data import build_dataloaders
from src.utils.segmentation.general import set_segmentation_input_channels
from src.scripts.segmentation.preprocess import *

from model import Model
import architecture as arch

"""
  Fully unsupervised clustering for segmentation ("IIC" = "IID").
  Train and test script.
  Network has two heads, for overclustering and final clustering.
"""


def interface():
    # TODO precheck config info with assertions
    config_file = ConfigFile("config.json")
    config = config_file.segmentation

    assert config.training.validation_mode == "IID"
    assert (
        config.architecture.head_A_class_count >= config.architecture.head_B_class_count
    )
    config.architecture.num_classes = config.architecture.head_B_class_count

    if "last_epoch" not in config:
        config.last_epoch = 0

    config.out_dir = str(Path(config.output.root).resolve() / str(config.dataset.id))
    out_dir = Path(config.out_dir).resolve()
    if not (out_dir.is_dir() and out_dir.exists()):
        out_dir.mkdir(parents=True, exist_ok=True)

    config.dataloader_batch_size = int(
        config.training.batch_size / config.training.num_dataloaders
    )
    config.output_k = config.architecture.head_B_class_count  # for eval code
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
    set_segmentation_input_channels(config)
    trunk = arch.VGGTrunk(
        structure=arch.STRUCTURE,
        input_channels=config.in_channels,
        **config.architecture.trunk
    )
    head_A = arch.SegmentationNet10aHead(
        feature_count=arch.STRUCTURE[-1][0],
        input_size=config.dataset.parameters.input_size,
        class_count=config.architecture.head_A_class_count,
        subhead_count=config.architecture.subhead_count,
    )
    head_B = arch.SegmentationNet10aHead(
        feature_count=arch.STRUCTURE[-1][0],
        input_size=config.dataset.parameters.input_size,
        class_count=config.architecture.head_B_class_count,
        subhead_count=config.architecture.subhead_count,
    )
    architecture = arch.SegmentationNet10aTwoHead(
        trunk=trunk, head_A=head_A, head_B=head_B
    )

    model = Model(config, net=architecture, dataloaders=dataloaders)
    model.train()


import shutil

if __name__ == "__main__":
    out = PurePath("out")
    shutil.rmtree(out, ignore_errors=True)
    interface()
