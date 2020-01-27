
from pathlib import Path


# Some path utilities
def ls(path):
    return [p for p in path.iterdir()]


def find_files(path, pattern=None):
    """ Recursively find all files """
    all_files = list()
    if path.is_dir():
        all_files.extend([f for p in ls(path) for f in find_files(p)])
    else:
        all_files.append(path)
    if pattern is not None:
        all_files = [f for f in all_files if f.match(pattern)]
    return all_files


# OK, let's do some duck-typing
Path.ls = ls
Path.find_files = find_files
