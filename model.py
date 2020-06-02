from typing import Dict, Any
from datetime import datetime
from pathlib import PurePath

import torch.nn
import numpy as np

from src.utils.segmentation.IID_losses import (
    IID_segmentation_loss,
    IID_segmentation_loss_uncollapsed,
)
from src.utils.segmentation.segmentation_eval import segmentation_eval
from src.scripts.segmentation.utils import (
    BatchStatistics,
    EpochStatistics,
    transfer_images,
    process,
    compute_losses,
    Canvas,
    StateFiles,
)


class Model:
    def __init__(self, config, net, dataloaders: Dict[str, Any]):
        # INITIALIZE
        state_files = StateFiles(config)
        net = torch.nn.DataParallel(net)
        net.train()
        optimizer = torch.optim.Adam(
            net.module.parameters(), lr=config.optimizer.learning_rate
        )
        stats = EpochStatistics()
        canvas = Canvas()
        num_epochs = config.training.num_epochs
        config.starting_epoch = 0

        # IF RESTART...
        if state_files.exists("config_binary", "latest"):
            config = state_files.load("config_binary", "latest")
            pytorch_data = state_files.load("pytorch", "latest")
            net.load_state_dict(pytorch_data["net"])
            optimizer.load_state_dict(pytorch_data["optimizer"])
            stats = state_files.load("statistics", "latest")
            config.num_epochs = num_epochs

        # GET LOSS FN
        # TODO put in function
        if not config.training.loss.use_uncollapsed:
            loss_fn = IID_segmentation_loss
        else:
            loss_fn = IID_segmentation_loss_uncollapsed

        # PREPARE HEAD, LAMB, DATALOADERS
        head_order = [h for h in config.training.head_order]
        required = [h for h in ["A", "B"]]
        assert set(required) == set(head_order)
        lambs = config.training.loss.lambs

        self._config = config
        self._heads = head_order
        self._lambs = lambs
        self._loaders = dataloaders
        self._state_files = state_files
        self._net = net
        self._optimizer = optimizer
        self._stats = stats
        self._canvas = canvas
        self._loss_fn = loss_fn
        self._dataloaders = dataloaders

    def train(self):
        epoch_stats = EpochStatistics()
        for epoch_number in range(
            self._config.starting_epoch, self._config.training.num_epochs
        ):
            print("Starting epoch: {epoch:4d}".format(epoch=epoch_number + 1))

            # PREPARE
            if epoch_number in self._config.optimizer.schedule:
                self._update_learning_rate(lr_mult=self._config.optimizer.multiplier)

            # PROCESS
            batch_stats = self._process_epoch()

            # EVALUATE
            eval_stats = segmentation_eval(
                self._config,
                self._net,
                mapping_assignment_dataloader=self._dataloaders["map_assign"],
                mapping_test_dataloader=self._dataloaders["map_test"],
                sobel=(self._config.preprocessor.sobelize),
                using_IR=self._config.dataset.parameters.use_ir,
            )
            if "acc" in epoch_stats:
                is_best = eval_stats["best"] > max(epoch_stats["acc"])
            else:
                is_best = True
            epoch_stats.add(
                {
                    "epoch": epoch_number,
                    "is_best": is_best,
                    "acc": eval_stats["best"],
                    "avg_subhead_acc": eval_stats["avg"],
                },
                batch_stats,
            )

            # SAVE
            self._net.module.cpu()
            save_dict = {
                "net": self._net.module.state_dict(),
                "optimizer": self._optimizer.state_dict(),
            }
            self._config.starting_epoch = epoch_number
            self._state_files.save_state("latest", self._config, save_dict, epoch_stats)
            if is_best:
                self._state_files.save_state(
                    "best", self._config, save_dict, epoch_stats
                )
            self._net.module.cuda()

            # DRAW
            self._canvas.draw(epoch_stats)
            name = PurePath(self._config.output.plot_name)
            if not name.suffix:
                name = name.with_suffix(".png")
            self._canvas.save(PurePath(self._config.out_dir) / name)

    def _process_epoch(self):
        batch_stats = {h: BatchStatistics() for h in self._heads}
        for head in self._heads:
            self._process_head(head, batch_stats[head])
        return batch_stats

    def _process_head(self, head: str, batch_stats: dict):
        batch_number = 0
        for data in zip(*self._dataloaders[head]):
            if batch_number % self._config.output.batch_print_freq == 0:
                verbose = True
            else:
                verbose = False
            process_fn = lambda x: process(x, self._net, head)
            prefix = " ".join(["{head:s}", "({batch:4d})"])
            self._process_batch(
                data, process_fn, self._lambs[head], batch_stats, verbose, prefix
            )
            batch_number += 1

    def _process_batch(
        self,
        data: tuple,
        process_fn,
        lamb: float,
        batch_stats: BatchStatistics,
        verbose: bool = True,
        message_prefix: str = "",
    ):
        self._net.module.zero_grad()
        images = transfer_images(data, self._config)
        outs = process_fn(images)
        losses = compute_losses(self._config, self._loss_fn, lamb, images, outs)
        loss = losses[0].item()
        loss_no_lamb = losses[1].item()

        if verbose:
            update = (" " * 2) + ", ".join(
                [
                    "{prefix:s}",
                    "loss: {loss:.6f}",
                    "loss no lamb: {loss_no_lamb:.6f}",
                    "time: {time:s}",
                ]
            )
            update.format(
                prefix=message_prefix,
                loss=loss,
                loss_no_lamb=loss_no_lamb,
                time=datetime.now(),
            )

        # TODO This could possible be more graceful...
        if not np.isfinite(loss):
            print("Loss is not finite... %s:" % str(losses[0]))
            exit(1)

        batch_stats.add({"loss": loss, "loss_no_lamb": loss_no_lamb})

        losses[0].backward()
        self._optimizer.step()

    def _update_learning_rate(self, lr_mult=0.1):
        for param_group in self._optimizer.param_groups:
            param_group["lr"] *= lr_mult

    @property
    def heads(self):
        return self._config.head_order

    @property
    def head_count(self):
        return len(self._config.head_order)
