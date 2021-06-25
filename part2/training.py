import argparse
import sys
from utils.log_config import logging

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
        print(f"Starting {type(self).__name__}, {self.cli_args}")


if __name__ == '__main__':
    LunaTrainingApp().main()
    logging.info("heyyy")
