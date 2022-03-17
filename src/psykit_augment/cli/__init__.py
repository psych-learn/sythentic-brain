import argparse


def add_version_parser(subparsers, formatter_class):
    from psykit_augment.__about__ import __version__
    subparser = subparsers.add_parser("version",formatter_class=formatter_class, help="Print the package version")
    subparser.set_defaults(func=lambda _: print(__version__))
    return subparser


class Formatter(argparse.ArgumentDefaultsHelpFormatter):
    def __init__(self, prog):
        super(Formatter, self).__init__(prog, max_help_position=35, width=150)


def get_arg_parser():
    # Initialising the parser with the description and usage
    arg_parser = argparse.ArgumentParser(
        description="psykit_augment", prog="python -m psykit-augment.cli", formatter_class=Formatter)
    arg_parser.add_argument("-v", "--version", action="store_true", help="Print package version")
    subparsers = arg_parser.add_subparsers(
        title="available commands", metavar="command [options ...]")
    # To parse version sub-command
    add_version_parser(subparsers, formatter_class=Formatter)
    return arg_parser


def run():
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        arg_parser.print_help()
        exit(1)
