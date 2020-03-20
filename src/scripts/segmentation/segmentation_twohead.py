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

from utils import (
    BatchStatistics,
    EpochStatistics,
    transfer_images,
    sobelize,
    process,
    compute_losses,
    Canvas,
    StateFiles,
)

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

    assert config.mode == "IID"
    assert "TwoHead" in config.arch
    assert config.output_k_A >= config.gt_k
    assert config.output_k_B == config.gt_k

    if "last_epoch" not in config.__dict__:
        config.last_epoch = 0
    config.out_dir = str(Path(config.out_root).resolve() / str(config.model_ind))
    config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)
    config.output_k = config.output_k_B  # for eval code
    config.use_doersch_datasets = False
    config.eval_mode = "hung"
    # TODO better mechanism for identifying number of channels in data
    # TODO more robust transform into desired number of channels
    set_segmentation_input_channels(config)

    out_dir = Path(config.out_dir).resolve()
    if not (out_dir.is_dir() and out_dir.exists()):
        out_dir.mkdir(parents=True, exist_ok=True)

    train(config)


def train(config):
    # SETUP
    # ! class MODEL
    state_files = StateFiles(config)
    num_epochs = config.num_epochs
    if state_files.exists("config_binary", "latest"):
        config = state_files.load("config_binary", "latest")
        config.restart = True
    else:
        config.restart = False
    config.num_epochs = num_epochs

    # DATALOADERS
    # TODO rework dataloader into general concept using map-based torch.utils.data.Dataloader
    dataloaders_head_A, mapping_assignment_dataloader, mapping_test_dataloader = segmentation_create_dataloaders(
        config
    )
    dataloaders_head_B = dataloaders_head_A  # unlike for clustering datasets

    # ARCHITECTURE
    # ! class MODEL
    net = archs.__dict__[config.arch](config)  # type: ignore
    pytorch_data = None
    if config.restart:
        pytorch_data = state_files.load("pytorch", "latest")
        net.load_state_dict(pytorch_data["net"])
    net.cuda()
    net = torch.nn.DataParallel(net)

    net.train()

    # OPTIMIZER
    # ! class MODEL
    optimizer = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
    if config.restart:
        assert pytorch_data is not None
        optimizer.load_state_dict(pytorch_data["optimizer"])

    # HEADS
    # ! class MODEL
    heads = ["A", "B"]
    if hasattr(config, "head_B_first") and config.head_B_first:
        heads = ["B", "A"]

    # STATISTICS
    if config.restart:
        next_epoch = config.last_epoch + 1
        epoch_stats = state_files.load("statistics", "latest")
    else:
        next_epoch = 1
        epoch_stats = EpochStatistics()
    print("Starting from epoch {epoch:d}".format(epoch=next_epoch))

    # CANVAS
    canvas = Canvas()

    # LOSS
    if not config.use_uncollapsed_loss:
        loss_fn = IID_segmentation_loss
    else:
        loss_fn = IID_segmentation_loss_uncollapsed

    # TRAIN
    # EPOCH LOOP
    for e_i in range(next_epoch, config.num_epochs + 1):
        print("Starting e_i: %d %s" % (e_i, datetime.now()))
        sys.stdout.flush()

        if e_i in config.lr_schedule:
            optimizer = update_lr(optimizer, lr_mult=config.lr_mult)

        # HEAD LOOP
        # ! class MODEL
        batch_stats = {h: BatchStatistics() for h in heads}
        for head in heads:
            if head.casefold() == "A".casefold():
                dataloaders = dataloaders_head_A
                lamb = config.lamb_A
            elif head.casefold() == "B".casefold():
                dataloaders = dataloaders_head_B
                lamb = config.lamb_B
            else:
                assert False

            iterators = (d for d in dataloaders)
            b_i = 0

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

                b_i += 1
                batch_stats[head].add(
                    {"loss": losses[0].item(), "loss_no_lamb": losses[1].item()}
                )

                # TORCH
                losses[0].backward()
                optimizer.step()

                # ! explicit del required if not in function scope
                del images
                del loaders

        # EVALUATE
        eval_stats = segmentation_eval(
            config,
            epoch_stats,
            net,
            mapping_assignment_dataloader=mapping_assignment_dataloader,
            mapping_test_dataloader=mapping_test_dataloader,
            sobel=(not config.no_sobel),
            using_IR=config.using_IR,
        )

        epoch_stats.add(
            {
                "epoch": e_i,
                "is_best": eval_stats["is_best"],
                "acc": eval_stats["best"],
                "avg_subhead_acc": eval_stats["avg"],
            },
            batch_stats,
        )

        # SAVE
        net.module.cpu()
        save_dict = {
            "net": net.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        config.last_epoch = e_i
        state_files.save_state("latest", config, save_dict, epoch_stats)
        if eval_stats["is_best"]:
            state_files.save_state("best", config, save_dict, epoch_stats)
        net.module.cuda()

        # UPDATE CANVAS
        canvas.draw(epoch_stats)
        name = PurePath(config.plot_name)
        if not name.suffix:
            name = name.with_suffix(".png")
        canvas.save(PurePath(config.out_dir) / name)

        # HACK DEBUGGING
        if config.test_code:
            exit(0)


import shutil

if __name__ == "__main__":
    out = PurePath("out/555")
    shutil.rmtree(out, ignore_errors=True)
    interface()
