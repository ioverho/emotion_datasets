import typing
import dataclasses
import pathlib
import logging


from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DatasetMetadata,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEBUG_METADATA = DatasetMetadata(
    description="A processor only used for debugging purposes",
    citation="",
    homepage="",
    license="",
    emotions=[],
    multilabel=False,
    continuous=False,
    system="",
    domain="",
)


@dataclasses.dataclass
class DebugProcessor(DatasetBase):
    name: str = "Debug"

    metadata: typing.ClassVar[DatasetMetadata] = DEBUG_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> None:
        return None

    def process_files(
        self,
        download_result: None,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> None:
        return None

    def teardown(
        self,
        download_result: None,
        processing_result: None,
        storage_options: dict,
    ) -> None:
        return None
