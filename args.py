import argparse
from pathlib import PurePath


class Arguments:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Provide the config file as json.")
        parser.add_argument(
            "-c", "--config", metavar="C", nargs="?", type=str, default="config.json"
        )
        args = parser.parse_args()

        self._args = args

    @property
    def config_file_path(self):
        return PurePath(self._args.config)
