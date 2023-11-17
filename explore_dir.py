import argparse

from memory import *


def is_int(s: str) -> bool:
    """
    Checks if the input can be converted to an integer.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string can be converted to an integer, False otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def explore_dir(root_dir: str, token_limit: int = 1000) -> None:
    """
    Navigates and manages the exploration of directories based on user input.

    This function initializes with a root directory and allows the user to navigate
    its subdirectories. It maintains a size limit for open directories (measured in tokens),
    and prompts the user for commands to open or close directories, or to exit the exploration.

    Args:
        root_dir (str): The root directory to start exploration from.
        token_limit (int, optional): The maximum size (in tokens) of directories that can be kept open.
                                     Defaults to 1000 if not specified.

    The function provides an interactive command-line interface for the user to:
    - Open a directory (by name or index).
    - Close a directory (by name).
    - View the current state of opened directories.
    - Exit the directory exploration.
    """
    root = Directory(root_dir)
    root.open(root_dir)

    while True:
        while root.size() > token_limit:
            root.close_oldest()

        print(root.to_str())

        ndir = input("> ")
        if ndir.startswith("open"):
            root.open(ndir[5:])
        elif ndir.startswith("close"):
            root.close(ndir[6:])
        elif ndir.startswith("exit"):
            break
        elif is_int(ndir):
            root.open_by_index(int(ndir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore directories.")
    parser.add_argument(
        "root_dir", type=str, help="Root directory to start exploration"
    )
    parser.add_argument(
        "-t",
        "--token_limit",
        type=int,
        default=1000,
        help="Token limit for directory size (default: 1000)",
    )

    args = parser.parse_args()
    explore_dir(args.root_dir, args.token_limit)
