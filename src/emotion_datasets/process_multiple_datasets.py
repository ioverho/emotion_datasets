import dataclasses
import typing
import logging

import hydra
import hydra.core
import hydra.core.config_store
import omegaconf

import emotion_datasets
from emotion_datasets.dataset_processing.base import get_dataset, DATASET_REGISTRY
from emotion_datasets.utils.manifest import get_manifest
from emotion_datasets.utils.config import ConfigBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class MultipleDatasetProcessingConfig(ConfigBase):
    datasets: typing.Dict[str, typing.Any] = dataclasses.field(
        default_factory=lambda: {
            dataset_name.lower(): dataset_class
            for dataset_name, dataset_class in DATASET_REGISTRY.items()
            if "debug" not in dataset_name
        }
    )

    skip: typing.List[str] = dataclasses.field(default_factory=lambda: [])

    skip_errors: bool = True

    print_config: bool = True


config_store = hydra.core.config_store.ConfigStore.instance()

config_store.store(
    name="process_multiple_datasets", node=MultipleDatasetProcessingConfig
)


@hydra.main(
    version_base=None, config_path=None, config_name="process_multiple_datasets"
)
def process_multiple_datasets(config: omegaconf.DictConfig):
    # Print the config for the user
    if config.print_config or config.debug:
        config_str = f"\n{'=' * 80}\nPARSED CONFIG:\n\n{omegaconf.OmegaConf.to_yaml(config, resolve=True)}\n{'=' * 80}"

        logger.info(config_str)

    # First process all the configs to make sure all scripts are correctly configured
    dataset_to_config = dict()
    for dataset_name, dataset_config in config.datasets.items():
        if dataset_name in config.skip:
            logger.debug(f"Orchestration - Skipping dataset {dataset_name}")
            continue

        wrapped_dataset_config = (
            emotion_datasets.process_one_dataset.DatasetProcessingConfig(
                dataset=dataset_config,
                file_system=config.file_system,
                huggingface=config.huggingface,
                overwrite=config.overwrite,
                # If print_config is True, only prints the full config once
                print_config=False,
                debug=config.debug,
            )
        )

        # Check to see if the config works
        try:
            get_dataset(**wrapped_dataset_config.dataset)
        except Exception as e:
            raise ValueError(
                f"Could not parse configuration for `datasets.{dataset_name}`. Encountered the following exception: {e}"
            )

        # Wrap the config
        dataset_to_config[dataset_name] = wrapped_dataset_config

    logger.info(
        f"Orchestration - Parsed configuration files for: {list(dataset_to_config.keys())}"
    )

    # Then run each script one-by-one
    for dataset_name, dataset_config in dataset_to_config.items():
        logger.info(f"Orchestration - Starting processing for dataset: {dataset_name}")

        try:
            emotion_datasets.process_one_dataset.process_dataset(dataset_config)
        except Exception as e:
            if config.skip_errors or config.debug:
                logger.critical(
                    f"Could not process dataset '{dataset_name}'. Encountered the following exception: {e}"
                )
            else:
                raise ValueError(
                    f"Could not process dataset '{dataset_name}'. Encountered the following exception: {e}"
                )

    logger.info("Orchestration - Finished processing all datasets")

    logger.info(get_manifest(data_dir=config.file_system.data_dir))


if __name__ == "__main__":
    process_multiple_datasets()
