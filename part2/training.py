# import argparse
import sys
import os
import argparse

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from utils.log_config import logging  # noqa
from utils.dsets import getCandidateInfo, getCt, LunaDataset  # noqa


# print(sys.path)
log = logging.getLogger(__name__)

log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


class LunaTrainingApp:
    def __init__(self, sys_argv=None) -> None:
        if sys_argv is None:
            # get arguments, excluding the file name
            sys_argv = sys.argv[1:]
        # define a parser for the arguments collected above
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--num-workers', help='Number of workers for data loading', default=8, type=int)
        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")


if __name__ == '__main__':
    LunaTrainingApp().main()
