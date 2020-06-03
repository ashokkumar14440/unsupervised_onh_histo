from pathlib import Path, PurePath

from inc.config_snake.config import ConfigFile, ConfigDict

from model import Model
import architecture as arch
import data
import preprocessing as pre
import utils

import cocostuff

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

    config.output_k = config.architecture.head_B_class_count  # for eval code
    config.eval_mode = "hung"

    train(config)


def train(config):
    LABEL_FILTERS = {"CocoFewLabels": cocostuff.CocoFewLabels}
    transformation = pre.Transformation(**config.transformations)

    if config.dataset.label_filter.name in LABEL_FILTERS:
        label_filter = LABEL_FILTERS[config.dataset.label_filter.name](
            class_count=config.architecture.num_classes,
            **config.dataset.label_filter.parameters
        )
        label_mapper = pre.LabelMapper(mapping_function=label_filter.apply)
    else:
        print("unable to find label mapper, using identity mapping")
        label_mapper = pre.LabelMapper

    image_info = utils.ImageInfo(**config.dataset.parameters)

    output_root = PurePath(config.output.root) / str(config.dataset.id)
    output_root = PurePath(Path(output_root).resolve())
    if not (Path(output_root).is_dir() and Path(output_root).exists()):
        Path(output_root).mkdir(parents=True, exist_ok=True)
    output_files = utils.OutputFiles(
        root_path=output_root,
        render_subfolder=config.output.rendering.folder,
        image_info=image_info,
    )

    preprocessing = pre.Preprocessing(
        transformation=transformation,
        image_info=image_info,
        label_mapper=label_mapper,
        **config.preprocessor
    )

    dataset = PurePath(config.dataset.root)
    if "partitions" in config.dataset:
        partitions = config.dataset.partitions
        image_folder = dataset / partitions.image
        label_folder = dataset / partitions.label
    else:
        image_folder = dataset
        label_folder = None
    EXTENSIONS = [".png", ".jpg"]

    train_prep = pre.TrainImagePreprocessor(
        image_info=image_info,
        preprocessing=preprocessing,
        output_files=output_files,
        do_render=config.output.rendering.enabled,
        render_limit=config.output.rendering.limit,
    )
    train_dataset = data.ImageFolderDataset(
        image_folder=image_folder,
        preprocessor=train_prep,
        extensions=EXTENSIONS,
        label_folder=label_folder,
    )
    train_dataloader = data.TrainDataLoader(
        dataset=train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
    )

    test_prep = pre.TestImagePreprocessor(
        image_info=image_info,
        preprocessing=preprocessing,
        output_files=output_files,
        do_render=config.output.rendering.enabled,
        render_limit=config.output.rendering.limit,
    )
    test_dataset = data.ImageFolderDataset(
        image_folder=image_folder,
        preprocessor=test_prep,
        extensions=EXTENSIONS,
        label_folder=label_folder,
    )
    test_dataloader = data.TestDataLoader(
        dataset=test_dataset, batch_size=config.training.batch_size
    )

    dataloaders = {
        "A": train_dataloader,
        "B": train_dataloader,
        "map_assign": test_dataloader,
        "map_test": test_dataloader,
    }
    trunk = arch.VGGTrunk(
        structure=arch.STRUCTURE,
        input_channels=image_info.channel_count,
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

    model = Model(
        config, output_root=output_root, net=architecture, dataloaders=dataloaders
    )
    model.train()


import shutil

if __name__ == "__main__":
    out = PurePath("out")
    shutil.rmtree(out, ignore_errors=True)
    interface()
