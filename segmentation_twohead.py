from pathlib import Path, PurePath
import shutil

import torch

import architecture as arch
import cocostuff
import data
from inc.config_snake.config import ConfigFile
import loss
from model import Model
import preprocessing as pre
import utils


def interface():
    config_file = ConfigFile("config.json")
    config = config_file.segmentation

    assert config.training.validation_mode == "IID"
    assert config.training.eval_mode == "hung"

    train(config)


def train(config):
    # INPUT IMAGE INFORMATION
    image_info = utils.ImageInfo(**config.dataset.parameters)

    # ARCH HEAD INFORMATION
    heads_info = arch.HeadsInfo(
        heads_info=config.architecture.heads.info,
        input_size=config.dataset.parameters.input_size,
        subhead_count=config.architecture.heads.subhead_count,
    )

    # OUTPUT_FILES
    output_root = PurePath(config.output.root) / str(config.dataset.id)
    output_root = PurePath(Path(output_root).resolve())
    if not (Path(output_root).is_dir() and Path(output_root).exists()):
        Path(output_root).mkdir(parents=True, exist_ok=True)
    output_files = utils.OutputFiles(root_path=output_root, image_info=image_info)

    # NETWORK ARCHITECTURE
    structure = arch.Structure(
        input_channels=image_info.channel_count,
        structure=config.architecture.trunk.structure,
    )
    trunk = arch.VGGTrunk(structure=structure, **config.architecture.trunk.parameters)
    net = arch.SegmentationNet10aTwoHead(
        trunk=trunk, heads=heads_info.build_heads(trunk.feature_count)
    )
    net.to(torch.device("cuda:0"))

    # OPTIMIZER
    optimizer = torch.optim.Adam(net.parameters(), lr=config.optimizer.learning_rate)

    # MODEL
    state_folder = output_files.get_sub_root(output_files.STATE)
    if Model.exists(state_folder):
        # LOAD EXISTING
        model = Model.load(state_folder, net=net, optimizer=optimizer)
    else:
        # CREATE NEW
        # LOSS
        iid_loss = loss.IIDLoss(
            heads=heads_info.order,
            output_files=output_files,
            do_render=config.output.rendering.enabled,
            **config.training.loss
        )

        # STATISTICS
        epoch_stats = utils.EpochStatistics(
            limit=config.training.num_epochs, output_files=output_files
        )

        model = Model(
            state_folder=state_folder,
            heads_info=heads_info,
            net=net,
            optimizer=optimizer,
            loss_fn=iid_loss,
            epoch_statistics=epoch_stats,
        )

    # PREPROCESSING
    # transformation
    transformation = pre.Transformation(**config.transformations)

    # label mapping
    LABEL_FILTERS = {"CocoFewLabels": cocostuff.CocoFewLabels}
    if config.dataset.label_filter.name in LABEL_FILTERS:
        label_filter = LABEL_FILTERS[config.dataset.label_filter.name](
            class_count=heads_info.class_count, **config.dataset.label_filter.parameters
        )
        label_mapper = pre.LabelMapper(mapping_function=label_filter.apply)
    else:
        print("unable to find label mapper, using identity mapping")
        label_mapper = pre.LabelMapper()

    # general preprocessing
    preprocessing = pre.Preprocessing(
        transformation=transformation,
        image_info=image_info,
        label_mapper=label_mapper,
        **config.preprocessor
    )

    # RENDERING PATHS
    # TODO into output_files
    dataset = PurePath(config.dataset.root)
    if "partitions" in config.dataset:
        partitions = config.dataset.partitions
        image_folder = dataset / partitions.image
        label_folder = dataset / partitions.label
    else:
        image_folder = dataset
        label_folder = None
    EXTENSIONS = [".png", ".jpg"]

    # TRAIN DATALOADER
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

    # TEST DATALOADER
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

    # DATALOADERS
    # TODO link this to arch heads
    dataloaders = {
        "A": train_dataloader,
        "B": train_dataloader,
        "map_assign": test_dataloader,
        "map_test": test_dataloader,
    }

    model.train(loaders=dataloaders)


if __name__ == "__main__":
    out = PurePath("out")
    shutil.rmtree(out, ignore_errors=True)
    interface()
