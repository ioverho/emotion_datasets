import pathlib
import json
import typing
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_manifest(
    data_subdir: pathlib.Path,
    dataset_name: str,
    dataset_info: typing.Dict[str, typing.Any],
):
    data_dir = data_subdir.parent

    if (data_dir / "manifest.json").exists():
        with open(data_dir / "manifest.json", "r") as f:
            manifest = json.load(fp=f)

        logger.debug(msg="Manifest - Manifest file read")

    else:
        logger.debug("Manifest - Manifest file does not yet exist")
        manifest = dict()

    manifest[dataset_name] = dataset_info

    with open(data_dir / "manifest.json", "w") as f:
        json.dump(obj=manifest, fp=f, indent=2, sort_keys=True)

        logger.debug(msg="Manifest - Wrote updated manifest file to disk")


def get_manifest(
    data_dir: pathlib.Path,
) -> str:
    if (data_dir / "manifest.json").exists():
        with open(data_dir / "manifest.json", "r") as f:
            manifest = json.load(fp=f)

        logger.debug(msg="Manifest - Manifest file read")

    else:
        logger.debug("Manifest - Manifest file does not yet exist")
        manifest = dict()

    manifest_str = json.dumps(obj=manifest, indent=2, sort_keys=True)

    return manifest_str
