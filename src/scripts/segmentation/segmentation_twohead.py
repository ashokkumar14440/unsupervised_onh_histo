import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

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

from .utils import transfer_images, sobelize, process, compute_losses

"""
  Fully unsupervised clustering for segmentation ("IIC" = "IID").
  Train and test script.
  Network has two heads, for overclustering and final clustering.
"""

# Options ----------------------------------------------------------------------

config = ConfigFile("config.json")

# Setup ------------------------------------------------------------------------

config.out_dir = os.path.join(config.out_root, str(config.model_ind))
config.dataloader_batch_sz = int(config.batch_sz / config.num_dataloaders)
assert config.mode == "IID"
assert "TwoHead" in config.arch
assert config.output_k_B == config.gt_k
config.output_k = config.output_k_B  # for eval code
assert config.output_k_A >= config.gt_k  # sanity
config.use_doersch_datasets = False
config.eval_mode = "hung"
set_segmentation_input_channels(config)

if not os.path.exists(config.out_dir):
    os.makedirs(config.out_dir)

dict_name = None
if config.restart:
    config_name = "config.pickle"
    dict_name = "latest.pytorch"

    given_config = config
    reloaded_config_path = os.path.join(given_config.out_dir, config_name)
    print("Loading restarting config from: %s" % reloaded_config_path)
    with open(reloaded_config_path, "rb") as config_f:
        config = pickle.load(config_f)
    assert config.model_ind == given_config.model_ind
    config.restart = True

    # copy over new num_epochs and lr schedule
    config.num_epochs = given_config.num_epochs
    config.lr_schedule = given_config.lr_schedule
else:
    print("Given config: %s" % config_to_str(config))


# Model ------------------------------------------------------


def train():
    dataloaders_head_A, mapping_assignment_dataloader, mapping_test_dataloader = segmentation_create_dataloaders(
        config
    )
    dataloaders_head_B = dataloaders_head_A  # unlike for clustering datasets

    net = archs.__dict__[config.arch](config)  # type: ignore
    dict = None
    if config.restart:
        assert dict_name is not None
        dict = torch.load(
            os.path.join(config.out_dir, dict_name),
            map_location=lambda storage, loc: storage,
        )
        net.load_state_dict(dict["net"])
    net.cuda()
    net = torch.nn.DataParallel(net)
    net.train()

    optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
    if config.restart:
        assert dict is not None
        optimiser.load_state_dict(dict["optimiser"])

    heads = ["A", "B"]
    if hasattr(config, "head_B_first") and config.head_B_first:
        heads = ["B", "A"]

    # Results
    # ----------------------------------------------------------------------

    if config.restart:
        next_epoch = config.last_epoch + 1
        print("starting from epoch %d" % next_epoch)

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

    fig, axarr = plt.subplots(6, sharex=False, figsize=(20, 20))

    if not config.use_uncollapsed_loss:
        print("using condensed loss (default)")
        loss_fn = IID_segmentation_loss
    else:
        print("using uncollapsed loss!")
        loss_fn = IID_segmentation_loss_uncollapsed

    # Train
    # ------------------------------------------------------------------------

    for e_i in range(next_epoch, config.num_epochs):
        print("Starting e_i: %d %s" % (e_i, datetime.now()))
        sys.stdout.flush()

        print("Checking lr_schedule")
        if e_i in config.lr_schedule:
            print("  is in schedule!")
            optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

        print("Head loop")
        for head_i in range(2):
            head = heads[head_i]
            if head == "A":
                print("  Head A")
                dataloaders = dataloaders_head_A
                epoch_loss = config.epoch_loss_head_A
                epoch_loss_no_lamb = config.epoch_loss_no_lamb_head_A
                lamb = config.lamb_A
            elif head == "B":
                print("  Head B")
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

            # PROCESS EXAMPLES
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
                if b_i == 2 and config.test_code:
                    break

            assert avg_loss_count > 0
            avg_loss = float(avg_loss / avg_loss_count)
            avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

            epoch_loss.append(avg_loss)
            epoch_loss_no_lamb.append(avg_loss_no_lamb)

        # Eval
        # -----------------------------------------------------------------------

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

        axarr[0].clear()
        axarr[0].plot(config.epoch_acc)
        axarr[0].set_title("acc (best), top: %f" % max(config.epoch_acc))

        axarr[1].clear()
        axarr[1].plot(config.epoch_avg_subhead_acc)
        axarr[1].set_title("acc (avg), top: %f" % max(config.epoch_avg_subhead_acc))

        axarr[2].clear()
        axarr[2].plot(config.epoch_loss_head_A)
        axarr[2].set_title("Loss head A")

        axarr[3].clear()
        axarr[3].plot(config.epoch_loss_no_lamb_head_A)
        axarr[3].set_title("Loss no lamb head A")

        axarr[4].clear()
        axarr[4].plot(config.epoch_loss_head_B)
        axarr[4].set_title("Loss head B")

        axarr[5].clear()
        axarr[5].plot(config.epoch_loss_no_lamb_head_B)
        axarr[5].set_title("Loss no lamb head B")

        fig.canvas.draw_idle()
        fig.savefig(os.path.join(config.out_dir, "plots.png"))

        if is_best or (e_i % config.save_freq == 0):
            net.module.cpu()
            save_dict = {
                "net": net.module.state_dict(),
                "optimiser": optimiser.state_dict(),
            }

            if e_i % config.save_freq == 0:
                torch.save(save_dict, os.path.join(config.out_dir, "latest.pytorch"))
                config.last_epoch = e_i  # for last saved version

            if is_best:
                torch.save(save_dict, os.path.join(config.out_dir, "best.pytorch"))

                with open(
                    os.path.join(config.out_dir, "best_config.pickle"), "wb"
                ) as outfile:
                    pickle.dump(config, outfile)

                with open(
                    os.path.join(config.out_dir, "best_config.txt"), "w"
                ) as text_file:
                    text_file.write("%s" % config)

            net.module.cuda()

        with open(os.path.join(config.out_dir, "config.pickle"), "wb") as outfile:
            pickle.dump(config, outfile)

        with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
            text_file.write("%s" % config)

        if config.test_code:
            exit(0)


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
else:
    train()
