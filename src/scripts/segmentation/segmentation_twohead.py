import sys
from datetime import datetime
from pathlib import Path, PurePath
from types import SimpleNamespace

import numpy as np
import torch

from inc.config_snake.config import ConfigFile

import src.archs as archs
from src.utils.cluster.general import config_to_str, get_opt, update_lr, nice
from src.utils.segmentation.segmentation_eval import segmentation_eval
from src.utils.segmentation.IID_losses import (
    IID_segmentation_loss,
    IID_segmentation_loss_uncollapsed,
)
from src.utils.segmentation.data import segmentation_create_dataloaders
from src.utils.segmentation.general import set_segmentation_input_channels

from utils import transfer_images, sobelize, process, compute_losses, Canvas, StateFiles

"""
  Fully unsupervised clustering for segmentation ("IIC" = "IID").
  Train and test script.
  Network has two heads, for overclustering and final clustering.
"""

# Options ----------------------------------------------------------------------

config_file = ConfigFile("config.json")
config = config_file.segmentation
config = config.to_json()
config = SimpleNamespace(**config)

# Setup ------------------------------------------------------------------------

if "last_epoch" not in config.__dict__:
    config.last_epoch = 0

config.out_dir = str(Path(config.out_root).resolve() / str(config.model_ind))
config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)

assert config.mode == "IID"
assert "TwoHead" in config.arch
assert config.output_k_B == config.gt_k

config.output_k = config.output_k_B  # for eval code

assert config.output_k_A >= config.gt_k  # sanity

config.use_doersch_datasets = False
config.eval_mode = "hung"
set_segmentation_input_channels(config)

if not Path(config.out_dir).is_dir():
    Path(config.out_dir).mkdir(parents=True, exist_ok=True)

state_files = StateFiles(config)
if state_files.exists_config():
    config = state_files.load_config("latest")
    config.restart = True
else:
    config.restart = False

# Model ------------------------------------------------------


def train():
    # SETUP
    # DATALOADERS
    dataloaders_head_A, mapping_assignment_dataloader, mapping_test_dataloader = segmentation_create_dataloaders(
        config
    )
    dataloaders_head_B = dataloaders_head_A  # unlike for clustering datasets

    # ARCHITECTURE
    net = archs.__dict__[config.arch](config)  # type: ignore
    pytorch_data = None
    if config.restart:
        pytorch_data = state_files.load_pytorch("latest")
        net.load_state_dict(pytorch_data["net"])
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.train()

    # OPTIMIZER
    optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
    if config.restart:
        assert pytorch_data is not None
        optimiser.load_state_dict(pytorch_data["optimiser"])

    # HEADS
    heads = ["A", "B"]
    if hasattr(config, "head_B_first") and config.head_B_first:
        heads = ["B", "A"]

    # STATISTICS
    if config.restart:
        next_epoch = config.last_epoch + 1
        print("starting from epoch %d" % next_epoch)

        # TODO these may not be needed if we are always loading precisely the latest epoch
        config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
        config.epoch_avg_subhead_acc = config.epoch_avg_subhead_acc[:next_epoch]
        config.epoch_stats = config.epoch_stats[:next_epoch]

        config.epoch_loss_head_A = config.epoch_loss_head_A[: (next_epoch - 1)]
        config.epoch_loss_no_lamb_head_A = config.epoch_loss_no_lamb_head_A[
            : (next_epoch - 1)
        ]
        config.epoch_loss_head_B = config.epoch_loss_head_B[: (next_epoch - 1)]
        config.epoch_loss_no_lamb_head_B = config.epoch_loss_no_lamb_head_B[
            : (next_epoch - 1)
        ]
    else:
        config.epoch_acc = []
        config.epoch_avg_subhead_acc = []
        config.epoch_stats = []

        config.epoch_loss_head_A = []
        config.epoch_loss_no_lamb_head_A = []

        config.epoch_loss_head_B = []
        config.epoch_loss_no_lamb_head_B = []

        _ = segmentation_eval(
            config,
            net,
            mapping_assignment_dataloader=mapping_assignment_dataloader,
            mapping_test_dataloader=mapping_test_dataloader,
            sobel=(not config.no_sobel),
            using_IR=config.using_IR,
        )

        print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
        sys.stdout.flush()
        next_epoch = 1

    # CANVAS
    canvas = Canvas()

    # LOSS
    if not config.use_uncollapsed_loss:
        loss_fn = IID_segmentation_loss
    else:
        loss_fn = IID_segmentation_loss_uncollapsed

    # TRAIN
    # EPOCH LOOP
    for e_i in range(next_epoch, config.num_epochs):
        print("Starting e_i: %d %s" % (e_i, datetime.now()))
        sys.stdout.flush()

        if e_i in config.lr_schedule:
            optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

        # HEAD LOOP
        for head_i in range(2):
            head = heads[head_i]
            if head == "A":
                dataloaders = dataloaders_head_A
                epoch_loss = config.epoch_loss_head_A
                epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_A
                lamb = config.lamb_A
            elif head == "B":
                dataloaders = dataloaders_head_B
                epoch_loss = config.epoch_loss_head_B
                epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_B
                lamb = config.lamb_B
            else:
                assert False

            iterators = (d for d in dataloaders)
            b_i = 0
            avg_loss = 0.0  # over heads and head_epochs (and sub_heads)
            avg_loss_no_lamb = 0.0
            avg_loss_count = 0

            # BATCH LOOP
            for loaders in zip(*iterators):
                net.module.zero_grad()
                images = transfer_images(loaders, config)
                if not config.no_sobel:
                    images = sobelize(images, config)
                outs = process(images, net, head)
                losses = compute_losses(config, loss_fn, lamb, images, outs)

                if ((b_i % 1) == 0) or (e_i == next_epoch):
                    print(
                        "Model ind %d epoch %d head %s batch: %d avg loss %f avg loss no "
                        "lamb %f "
                        "time %s"
                        % (
                            config.model_ind,
                            e_i,
                            head,
                            b_i,
                            losses[0].item(),
                            losses[1].item(),
                            datetime.now(),
                        )
                    )
                    sys.stdout.flush()

                # TODO This could possible be more graceful...
                if not np.isfinite(losses[0].item()):
                    print("Loss is not finite... %s:" % str(losses[0]))
                    exit(1)

                avg_loss += losses[0].item()
                avg_loss_no_lamb += losses[1].item()
                avg_loss_count += 1

                losses[0].backward()
                optimiser.step()

                # ! explicit del required if not in function scope
                del images
                del loaders

                b_i += 1
                # HACK DEBUGGING
                if b_i == 2 and config.test_code:
                    break

            assert avg_loss_count > 0
            avg_loss = float(avg_loss / avg_loss_count)
            avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

            epoch_loss.append(avg_loss)
            epoch_loss_no_lamb.append(avg_loss_no_lamb)

        # EVALUATE
        is_best = segmentation_eval(
            config,
            net,
            mapping_assignment_dataloader=mapping_assignment_dataloader,
            mapping_test_dataloader=mapping_test_dataloader,
            sobel=(not config.no_sobel),
            using_IR=config.using_IR,
        )

        print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
        sys.stdout.flush()

        # SAVE
        do_save_latest = e_i % config.save_freq == 0
        if is_best or do_save_latest:
            net.module.cpu()
            save_dict = {
                "net": net.module.state_dict(),
                "optimiser": optimiser.state_dict(),
            }
            config.last_epoch = e_i
            if do_save_latest == 0:
                state_files.save_state("latest", config, save_dict)
            if is_best:
                state_files.save_state("best", config, save_dict)
            net.module.cuda()

        # UPDATE CANVAS
        canvas.draw(config)
        name = PurePath(config.plot_name)
        if not name.suffix:
            name = name.with_suffix(".png")
        canvas.save(PurePath(config.out_dir) / name)

        # HACK DEBUGGING
        if config.test_code:
            exit(0)


if __name__ == "__main__":
    train()
