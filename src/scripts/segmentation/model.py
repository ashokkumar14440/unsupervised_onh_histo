from typing import Dict

import torch.nn

import src.archs as archs
from src.utils.cluster.general import get_opt, update_lr
from utils import StateFiles


class Model:
    def __init__(self, config, dataloaders: Dict[str, list]):
        state_files = StateFiles(config)
        net = archs.__dict__[self._config.arch](self._config)  # type: ignore
        optimizer = get_opt(config.opt)(net.module.parameters(), lr=config.lr)

        # RESTART
        if state_files.exists_config("latest"):
            config = state_files.load_config("latest")
            pytorch_data = state_files.load_pytorch("latest")
            net.load_state_dict(pytorch_data["net"])
            optimizer.load_state_dict(pytorch_data["optimizer"])

        net.cuda()
        net = torch.nn.DataParallel(net)

        head_order = [h.casefold() for h in config.head_order]
        required = [h.casefold() for h in dataloaders.keys()]
        assert set(required) == set(head_order)

        self._config = config
        self._loaders = dataloaders
        self._state_files = state_files
        self._net = net
        self._optimizer = optimizer

    @property
    def heads(self):
        return self._config.head_order

    @property
    def head_count(self):
        return len(self._config.head_order)
