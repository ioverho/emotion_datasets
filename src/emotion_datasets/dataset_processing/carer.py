import dataclasses
import pathlib
import os
import logging
import typing
import shutil

import datasets

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadResult,
    ProcessingResult,
    DatasetMetadata,
)
from emotion_datasets.utils import (
    get_file_stats,
    update_manifest,
    update_bib_file,
    update_samples,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CARER_METADATA = DatasetMetadata(
    description="The CARER dataset, as processed using 'emotion_datasets'. CARER is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.",
    citation=(
        "@inproceedings{emotion_dataset_carer,"
        "\n title = '{CARER}: Contextualized Affect Representations for Emotion Recognition',"
        "\n author = 'Saravia, Elvis  and"
        "\n Liu, Hsien-Chi Toby  and"
        "\n Huang, Yen-Hao  and"
        "\n Wu, Junlin  and"
        "\n Chen, Yi-Shin',"
        "\n booktitle = 'Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing',"
        "\n month = oct # '-' # nov,"
        "\n year = '2018',"
        "\n address = 'Brussels, Belgium',"
        "\n publisher = 'Association for Computational Linguistics',"
        "\n url = 'https://www.aclweb.org/anthology/D18-1404',"
        "\n doi = '10.18653/v1/D18-1404',"
        "\n pages = '3687--3697',"
        "\n}"
    ),
    homepage="https://github.com/dair-ai/emotion_dataset",
    license="The dataset should be used for educational and research purposes only.",
    emotions=[],
    multilabel=False,
    continuous=False,
    system="Hashtags in Twitter posts corresponding to Ekman's core emotions",
    domain="Twitter posts",
)


@dataclasses.dataclass
class CARERDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path


@dataclasses.dataclass
class CARERProcessingResult(
    ProcessingResult,
):
    data_dir: pathlib.Path


@dataclasses.dataclass
class CARERProcessor(DatasetBase):
    name: str = "CARER"

    hf_repo: str = "dair-ai/emotion"
    hf_config_name: str = "split"
    hf_splits: str = "train+validation+test"

    label_map: typing.Dict[int, str] = dataclasses.field(
        default_factory=lambda: {
            0: "sadness",
            1: "joy",
            2: "love",
            3: "anger",
            4: "fear",
            5: "surprise",
        }
    )

    metadata: typing.ClassVar[DatasetMetadata] = CARER_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> CARERDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(name=downloads_subdir, exist_ok=True)

        logger.info(
            f"Download - Downloading dataset from HuggingFace Hub: {self.hf_repo}"
        )

        datasets.load_dataset(
            path=self.hf_repo,
            name=self.hf_config_name,
            cache_dir=str(downloads_subdir),
            split=self.hf_splits,
            keep_in_memory=False,
        )

        return CARERDownloadResult(downloads_subdir=downloads_subdir)

    def process_files(
        self,
        download_result: CARERDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> CARERProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        logger.info(
            f"Processing - Loading dataset from cache at: {download_result.downloads_subdir}"
        )

        hf_dataset: datasets.Dataset = datasets.load_dataset(
            path=self.hf_repo,
            name=self.hf_config_name,
            cache_dir=str(download_result.downloads_subdir),
            split=self.hf_splits,
            keep_in_memory=False,
        )  # type: ignore

        def map_label_to_emotions(row: dict):
            emotions = {
                "sadness": False,
                "joy": False,
                "love": False,
                "anger": False,
                "fear": False,
                "surprise": False,
            }

            emotion = self.label_map[row["label"]]

            emotions[emotion] = True

            return {**row, **emotions}

        mapped_dataset = hf_dataset.map(map_label_to_emotions)
        mapped_dataset = mapped_dataset.remove_columns(["label"])

        logger.info(f"Processing - Saving HuggingFace dataset: {data_subdir}")

        mapped_dataset.info.dataset_name = self.name
        mapped_dataset.info.description = self.metadata.description
        mapped_dataset.info.citation = self.metadata.citation
        mapped_dataset.info.homepage = self.metadata.homepage
        mapped_dataset.info.license = self.metadata.license

        mapped_dataset.save_to_disk(
            dataset_path=str(data_subdir),
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
            storage_options=storage_options,
        )

        logger.info("Processing - Finished saving HuggingFace dataset.")

        data_dir_summary = {
            "data": [],
        }
        for fp in data_subdir.glob("*"):
            file_stats = get_file_stats(fp=fp, data_dir=data_dir)

            data_dir_summary["data"].append(file_stats)

        update_manifest(
            data_subdir=data_subdir,
            dataset_name=self.name,
            dataset_info=data_dir_summary,
        )

        update_bib_file(
            data_subdir=data_subdir,
            dataset_metadata=self.metadata,
        )

        update_samples(
            data_subdir=data_subdir,
            dataset_name=self.name,
            storage_options=storage_options,
        )

        logger.info("Processing - Finished dataset processing.")

        logger.info(
            f"Processing - Find summary index file at: {data_subdir / 'index.json'}."
        )

        return CARERProcessingResult(data_dir=data_subdir)

    def teardown(
        self,
        download_result: CARERDownloadResult,
        processing_result: CARERProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_dir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
