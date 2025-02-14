from abc import ABCMeta
import pathlib
import logging
import shutil
import os

DATASET_REGISTRY = dict()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetBase(metaclass=ABCMeta):
    name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate =============================================================
        # Make sure that all aliases are unique
        if cls.name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{cls.name}' has already been registered.")

        # Register =============================================================
        DATASET_REGISTRY[cls.name] = cls

    @staticmethod
    def check_directory(data_subdir: pathlib.Path, overwrite: bool):
        if data_subdir.is_dir() and (any(data_subdir.iterdir())):
            logger.info(f"Filesystem - Found non-empty directory at {data_subdir}")
            if overwrite:
                logger.info(
                    "Filesystem - {data_subdir} already exists and is not empty. Overwrite set to True, will delete dir and create new dir."
                )
                shutil.rmtree(data_subdir)

            else:
                raise AssertionError(
                    f"Filesystem - {data_subdir} already exists and is not empty. Either set 'overwrite' to `True`, or have output written to a different directory using 'file_system.output_dir'"
                )
        else:
            os.makedirs(data_subdir, exist_ok=True)


class DownloadResult(metaclass=ABCMeta):
    pass


class ProcessingResult(metaclass=ABCMeta):
    pass


class DownloadError(Exception):
    pass


class DatasetProcessingError(Exception):
    pass


def get_dataset(name: str, **kwargs):
    try:
        return DATASET_REGISTRY[name](name=name, **kwargs)
    except KeyError:
        raise KeyError(
            f"Dataset `{name}` is not implemented. Please choose one of {set(DATASET_REGISTRY.keys())}"
        )
