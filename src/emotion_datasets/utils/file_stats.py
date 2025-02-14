import os
import pathlib
import typing


def get_file_stats(fp: pathlib.Path, data_dir: pathlib.Path) -> dict[str, typing.Any]:
    file_stats = os.stat(fp)

    stats = {
        "name": fp.name,
        "file_path": str(fp.relative_to(data_dir)),
        "file_size": file_stats.st_size,
        "creation_time": file_stats.st_mtime,
    }

    return stats
