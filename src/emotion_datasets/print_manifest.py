import logging
import typing
import dataclasses
import os
import pathlib

import hydra
import hydra.conf
import hydra.core.config_store

from emotion_datasets.utils import get_manifest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class FileSystemConfig:
    output_dir: str | pathlib.Path = "output"
    log_dir: str | pathlib.Path = "${file_system.output_dir}/logs"
    downloads_dir: str | pathlib.Path = "${file_system.output_dir}/downloads"
    data_dir: str | pathlib.Path = "${file_system.output_dir}/data"


@dataclasses.dataclass
class Config:
    defaults: typing.List[typing.Any] = dataclasses.field(
        default_factory=lambda: [
            "_self_",
            {"dataset": None},
        ]
    )

    file_system: FileSystemConfig = dataclasses.field(
        default_factory=lambda: FileSystemConfig
    )  # type: ignore

    debug: bool = False

    hydra: typing.Any = dataclasses.field(
        default_factory=lambda: {
            "job": {
                "name": "print_manifest",
                "chdir": False,
            },
            "run": {
                "dir": "${file_system.log_dir}/${hydra.job.name}/${now:%y%m%d %H%M%S}"
            },
            "verbose": "${debug}",
        }
    )


config_store = hydra.core.config_store.ConfigStore.instance()

config_store.store(name="print_manifest", node=Config)


@hydra.main(version_base=None, config_name="print_manifest")
def print_manifest(config: Config) -> None:
    # ==========================================================================
    # Setup
    # ==========================================================================
    # Filesystem ===============================================================
    config.file_system.downloads_dir = pathlib.Path(
        config.file_system.downloads_dir
    ).resolve()
    os.makedirs(config.file_system.downloads_dir, exist_ok=True)

    config.file_system.data_dir = pathlib.Path(config.file_system.data_dir).resolve()
    os.makedirs(config.file_system.data_dir, exist_ok=True)

    # ==========================================================================
    # Printing the manifest
    # ==========================================================================
    logger.info(get_manifest(data_dir=config.file_system.data_dir))


if __name__ == "__main__":
    print_manifest()
