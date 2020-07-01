import torch

import cocostuff
import data
from inc.config_snake.config import ConfigFile
import loss
from model import Model
import preprocessing as pre
import utils
import setup


def interface():
    config_file = ConfigFile("config.json")
    config = config_file.segmentation

    assert config.training.validation_mode == "IID"
    assert config.training.eval_mode == "hung"

    train(config)


def train(config):
    # SETUP
    components = setup.setup(config)
    image_info = components["image_info"]
    heads_info = components["heads_info"]
    output_files = components["output_files"]
    state_folder = components["state_folder"]
    image_folder = components["image_folder"]
    label_folder = components["label_folder"]
    net = components["net"]

    # FORCE RESTART
    if config.output.force_training_restart:
        output_files.clear_output()

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
    if (
        "label_filter" in config.dataset
        and config.dataset.label_filter.name in LABEL_FILTERS
    ):
        label_filter = LABEL_FILTERS[config.dataset.label_filter.name](
            class_count=heads_info.class_count, **config.dataset.label_filter.parameters
        )
        label_mapper = pre.LabelMapper(mapping_function=label_filter.apply)
    else:
        print("unable to find label mapper, using identity mapping")
        label_mapper = pre.LabelMapper()

    # general preprocessing
    preprocessing = pre.TransformPreprocessing(
        transformation=transformation,
        image_info=image_info,
        label_mapper=label_mapper,
        **config.preprocessor
    )

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
        extensions=config.dataset.extensions,
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
        extensions=config.dataset.extensions,
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
    interface()
