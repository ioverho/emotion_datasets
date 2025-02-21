import logging
import typing
import dataclasses
import os
import pathlib

import hydra
import hydra.conf
import hydra.core.config_store
import omegaconf

from emotion_datasets.dataset_processing import get_dataset
from emotion_datasets.dataset_processing.base import DATASET_REGISTRY
from emotion_datasets.utils.config import ConfigBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass(kw_only=True)
class DatasetProcessingConfig(ConfigBase):
    dataset: typing.Any

    defaults: typing.List[typing.Any] = dataclasses.field(
        default_factory=lambda: [
            "_self_",
            {"dataset": None},
        ]
    )

    hydra: typing.Any = dataclasses.field(
        default_factory=lambda: {
            "job": {
                "name": "dataset_processing",
                "chdir": False,
            },
            "run": {
                "dir": "${file_system.log_dir}/${hydra.job.name}/${dataset.name}/${now:%y%m%d %H%M%S}"
            },
            "verbose": "${debug}",
        }
    )


config_store = hydra.core.config_store.ConfigStore.instance()

for dataset_name, dataset_class in DATASET_REGISTRY.items():
    config_store.store(group="dataset", name=dataset_name.lower(), node=dataset_class)

config_store.store(name="process_dataset", node=DatasetProcessingConfig)


@hydra.main(version_base=None, config_name="process_dataset")
def process_dataset(config: DatasetProcessingConfig) -> None:
    # ==========================================================================
    # Validation
    # ==========================================================================
    try:
        config.dataset
    except omegaconf.errors.MissingMandatoryValue:
        raise omegaconf.errors.MissingMandatoryValue(
            f"CLI is missing a value for `dataset`. Please choose one of {set(DATASET_REGISTRY.keys())}."
        )

    # ==========================================================================
    # Setup
    # ==========================================================================
    # Print the dataset name
    dataset_title = ("  " + config.dataset.name + "  ").center(80, "=")
    title_str = f"\n{'=' * 80}\n{dataset_title}\n{'=' * 80}"

    logger.info(title_str)

    # Print the config for the user
    if config.print_config or config.debug:
        config_str = f"\n{'=' * 80}\nPARSED CONFIG:\n\n{omegaconf.OmegaConf.to_yaml(config, resolve=True)}\n{'=' * 80}"

        logger.info(config_str)

    # Filesystem ===============================================================
    config.file_system.downloads_dir = pathlib.Path(
        config.file_system.downloads_dir
    ).resolve()
    os.makedirs(config.file_system.downloads_dir, exist_ok=True)

    config.file_system.data_dir = pathlib.Path(config.file_system.data_dir).resolve()
    os.makedirs(config.file_system.data_dir, exist_ok=True)

    logger.info("Filesystem - Finished setting up the output directories")
    logger.info(f"Filesystem - Find downloads in {config.file_system.downloads_dir}")
    logger.info(f"Filesystem - Find processed data in {config.file_system.data_dir}")

    # ==========================================================================
    # Dataset Processing
    # ==========================================================================
    # Fetch the dataset class
    dataset = get_dataset(**config.dataset)
    logger.info(f"Setup - Fetched the dataset class for '{dataset.name}'")

    logger.info("Downloading - Starting download of files")

    # Download the dataset files
    download_result = dataset.download_files(
        downloads_dir=config.file_system.downloads_dir,
    )

    logger.info("Processing - Starting processing of files")

    # Process the dataset files
    processing_result = dataset.process_files(
        download_result=download_result,
        data_dir=config.file_system.data_dir,
        overwrite=config.overwrite,
        max_shard_size=config.huggingface.max_shard_size,
        num_shards=config.huggingface.num_shards,
        num_proc=config.huggingface.num_proc,
        storage_options=config.huggingface.storage_options,
    )

    logger.info("Teardown - Removing any unecessary files")

    # Delete any intermediate files and the downloaded files
    dataset.teardown(
        download_result=download_result,
        processing_result=processing_result,
        storage_options=config.huggingface.storage_options,
    )

    logger.info("Script finished succesfully")


if __name__ == "__main__":
    process_dataset()
