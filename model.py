from pathlib import Path, PurePath
from typing import Any, Dict, Union, Optional

import numpy as np
import torch
import loss
import utils

import architecture as arch
import data

PathLike = Union[str, Path, PurePath]

# canvas passed to epoch_stats
class Model:
    _ACC = "acc"
    _ACC_T = "Accuracy, Best"
    _AVG = "avg_subhead_acc"
    _AVG_T = "Accuracy, Average"
    _LOSS = "loss_{:s}"
    _LOSS_T = "Loss, Head {:s}"
    _LOSS_NL = "loss_no_lamb_{:s}"
    _LOSS_NL_T = "Loss, Head {:s}, No Lamb"
    _BEST_SUBHEAD = "best_subhead_{:s}"
    _BEST_SUBHEAD_LOSS = "best_subhead_loss_{:s}"

    _BASE_TITLES = {_ACC: _ACC_T, _AVG: _AVG_T}

    def __init__(
        self,
        state_folder: PathLike,
        heads_info: arch.HeadsInfo,
        net: torch.nn.Module,
        loss_fn: loss.Loss,
        epoch_statistics: utils.EpochStatistics,
        optimizer: Optional[torch.optim.Optimizer] = None,  # required for train()
    ):
        assert Path(state_folder).is_dir()
        assert Path(state_folder).exists()

        titles = self._BASE_TITLES.copy()
        for head in heads_info.order:
            titles[self._LOSS.format(head)] = self._LOSS_T.format(head)
            titles[self._LOSS_NL.format(head)] = self._LOSS_NL_T.format(head)

        self._state_folder = PurePath(state_folder)
        self._title_template = titles
        self._heads_info = heads_info
        self._net = self.parallelize(net)
        self._loss_fn = loss_fn
        self._epoch_stats = epoch_statistics
        self._opt = optimizer

    def train(self, loaders: Dict[str, torch.utils.data.DataLoader]):
        assert self._opt is not None
        for head in self._heads_info.order:
            assert head in loaders
        self._data = loaders

        self._net.train()
        for epoch in self._epoch_stats.epochs:
            print_epoch = epoch + 1
            print("epoch {:d}".format(print_epoch))
            # TODO update opt with lr schedule
            batch_stats = self._process_heads()
            primary_batch_means = batch_stats[self._heads_info.primary_head].get_means()
            primary_head_key = self._LOSS.format(self._heads_info.primary_head)
            primary_loss = primary_batch_means[primary_head_key]
            is_best = self._epoch_stats.is_smaller(
                key=primary_head_key, value=primary_loss
            )
            # TODO evaluate
            eval_stats = {"best": 0.0, "avg": 0.0}  # ! REMOVE ME
            epoch_values = {
                "epoch": print_epoch,
                "is_best": is_best,
                self._ACC: eval_stats["best"],
                self._AVG: eval_stats["avg"],
            }
            self._epoch_stats.add(
                values=epoch_values, batch_statistics=list(batch_stats.values())
            )
            self.save(is_best=is_best)
            self._epoch_stats.draw(titles=self._title_template)
            self._epoch_stats.save_data()
        self._net.eval()

    def evaluate(self, output_files: utils.OutputFiles, loader: data.EvalDataLoader):
        self._net.eval()
        with torch.no_grad():
            best = self._epoch_stats.best_epoch
            best_subhead = int(
                best[self._BEST_SUBHEAD.format(self._heads_info.primary_head)].item()
            )
            for data in loader:
                out = self._net(data, head=self._heads_info.primary_head)
                lbl = out["image"][best_subhead]
                lbl = lbl.detach().cpu().numpy()
                lbl = lbl.argmax(axis=1).astype(np.uint8)
                lbl = lbl[..., np.newaxis]
                lbl = loader.reassemble(
                    image=lbl, patch_count=out["patch_count"], padding=out["padding"]
                ).squeeze()
                name = PurePath(out["file_path"]).stem
                output_files.save_label(
                    name=name, label=lbl, subfolder=output_files.EVAL
                )
                img = data["image"]
                img = img.detach().cpu().numpy()
                img = img.transpose((0, 2, 3, 1))
                if img.ndim == 4 and img.shape[-1] > 1:
                    img = img[..., 0]
                    img = img[..., np.newaxis]
                img = loader.reassemble(
                    image=img, patch_count=data["patch_count"], padding=data["padding"]
                ).squeeze()
                output_files.save_rgb_label(
                    name=name, label=lbl, image=img, subfolder=output_files.EVAL
                )
                output_files.save_rgb_label(
                    name=name, label=lbl, subfolder=output_files.EVAL
                )

    def _process_heads(self):
        head_stats = {h: utils.BatchStatistics() for h in self._heads_info.order}
        for head in self._heads_info.order:
            print("  head {:s}".format(head))
            stats = head_stats[head]
            stats = self._process_batches(head, stats)
            head_stats[head] = stats
        return head_stats

    def _process_batches(self, head: str, stats: utils.BatchStatistics):
        loader = self._data[head]
        count = len(loader)
        for i, batch in enumerate(loader):
            print("    batch {:d}/{:d}".format(i + 1, count))
            data = self._process_images(head, batch)
            stats.add(data)
        return stats

    def _process_images(self, head: str, batch: Dict[str, Any]):
        self._net.module.zero_grad()
        processed = self._net(head=head, data=batch)
        loss_data = self._loss_fn(head=head, data=processed)
        out = {
            self._LOSS.format(head): loss_data["avg_loss"].item(),
            self._LOSS_NL.format(head): loss_data["avg_loss_no_lamb"].item(),
            self._BEST_SUBHEAD.format(head): loss_data["best_subhead"].item(),
            self._BEST_SUBHEAD_LOSS.format(head): loss_data["best_subhead_loss"].item(),
        }
        loss_data["avg_loss"].backward()
        self._opt.step()
        return out

    def _get_batch_count(self, head):
        return len(self._data[head])

    STATS_FILE = "stats.csv"
    LATEST_FILE = "latest.pickle"
    BEST_FILE = "best.pickle"
    LOSS_FILE = "loss.pickle"
    HEADS_FILE = "heads.pickle"
    FILES = [STATS_FILE, LATEST_FILE, BEST_FILE, LOSS_FILE, HEADS_FILE]

    def save(self, is_best: bool):
        state_folder = self._state_folder
        self._epoch_stats.save(state_folder / self.STATS_FILE)
        model = {
            "net": self._net.module.state_dict(),
            "optimizer": self._opt.state_dict(),
        }
        torch.save(model, str(state_folder / self.LATEST_FILE))
        if is_best:
            torch.save(model, str(state_folder / self.BEST_FILE))
        self._loss_fn.save(state_folder / self.LOSS_FILE)
        self._heads_info.save(state_folder / self.HEADS_FILE)

    @classmethod
    def load(
        cls,
        state_folder: PathLike,
        net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,  # required for train()
        use_best_net: bool = False,
    ):
        assert Path(state_folder).is_dir()
        assert Path(state_folder).exists()
        stats = utils.EpochStatistics.load(state_folder / cls.STATS_FILE)
        if use_best_net:
            model_file = cls.BEST_FILE
        else:
            model_file = cls.LATEST_FILE
        model = torch.load(str(state_folder / model_file))
        net.load_state_dict(model["net"])
        loss_fn = loss.Loss.load(state_folder / cls.LOSS_FILE)
        heads_info = arch.HeadsInfo.load(state_folder / cls.HEADS_FILE)
        if optimizer is not None:
            optimizer.load_state_dict(model["optimizer"])
        return cls(
            state_folder=state_folder,
            heads_info=heads_info,
            net=net,
            loss_fn=loss_fn,
            epoch_statistics=stats,
            optimizer=optimizer,
        )

    @staticmethod
    def exists(state_folder: PathLike):
        ok = Path(state_folder).is_dir()
        ok = ok and Path(state_folder).exists()
        for f in Model.FILES:
            path = state_folder / f
            ok_path = Path(path).is_file()
            ok_path = Path(path).exists()
            ok = ok and ok_path
        return ok

    @staticmethod
    def parallelize(net):
        return torch.nn.DataParallel(net)
