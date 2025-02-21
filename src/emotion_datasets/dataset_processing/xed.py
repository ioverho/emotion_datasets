import csv
import dataclasses
import json
import pathlib
import logging
import os
import shutil
import typing

import datasets

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadError,
    DownloadResult,
    ProcessingResult,
    DatasetMetadata,
)
from emotion_datasets.utils import download, get_file_stats, update_manifest, update_bib_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

XED_METADATA = DatasetMetadata(
    description="The XED dataset, as processed by 'emotion_datasets'. The dataset consists of emotion annotated movie subtitles from OPUS. Plutchik's 8 core emotions were used to annotate. The data is multilabel. The original annotations have been sourced for mainly English and Finnish. This is the English-only subset.",
    citation=(
        "@article{emotion_dataset_xed,"
        "\n  title={XED: A multilingual dataset for sentiment analysis and emotion detection},"
        '\n  author={{"O}hman, Emily and P{\'a}mies, Marc and Kajava, Kaisla and Tiedemann, J{"o}rg},'
        "\n  journal={arXiv preprint arXiv:2011.01612},"
        "\n  year={2020}"
        "\n}"
    ),
    homepage="https://github.com/Helsinki-NLP/XED/tree/master",
    license="",
    emotions=[
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
        "neutral",
    ],
    multilabel=True,
    continuous=True,
    system="Plutchik core emotions",
    domain="Subtitles",
)


@dataclasses.dataclass
class XEDDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    emotion_file_path: pathlib.Path
    neutral_file_path: pathlib.Path


@dataclasses.dataclass
class XEDProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path


@dataclasses.dataclass
class XEDProcessor(DatasetBase):
    name: str = "XED"

    emotion_url: str = "https://raw.githubusercontent.com/Helsinki-NLP/XED/refs/heads/master/AnnotatedData/en-annotated.tsv"
    neutral_url: str = "https://raw.githubusercontent.com/Helsinki-NLP/XED/refs/heads/master/AnnotatedData/neu_en.txt"

    emotion_map: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "1": "anger",
            "2": "anticipation",
            "3": "disgust",
            "4": "fear",
            "5": "joy",
            "6": "sadness",
            "7": "surprise",
            "8": "trust",
            "9": "neutral",
        }
    )

    metadata: typing.ClassVar[DatasetMetadata] = XED_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> XEDDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(name=downloads_subdir, exist_ok=True)

        emotion_file_name = pathlib.Path(self.emotion_url).name

        emotion_file_path = downloads_subdir / "emotion.tsv"

        logger.info(
            f"Download - Downloading file with emotional subtitles: {emotion_file_name}"
        )

        try:
            download(
                url=self.emotion_url,
                file_path=emotion_file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not file {emotion_file_name}. Encountered the following exception: {e}"
            )

        neutral_file_name = pathlib.Path(self.neutral_url).name

        neutral_file_path = downloads_subdir / "neutral.tsv"

        logger.info(
            f"Download - Downloading file with neutral subtitles: {neutral_file_name}"
        )

        try:
            download(
                url=self.neutral_url,
                file_path=neutral_file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download {neutral_file_name}. Encountered the following exception: {e}"
            )

        download_result = XEDDownloadResult(
            downloads_subdir=downloads_subdir,
            emotion_file_path=emotion_file_path,
            neutral_file_path=neutral_file_path,
        )

        return download_result

    def get_default_emotion_labels(self) -> dict[str, bool]:
        return {emotion: False for emotion in self.emotion_map.values()}

    def process_files(
        self,
        download_result: XEDDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: int | None,
        num_proc: int | None,
        storage_options: dict,
    ) -> XEDProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        records = []

        # Process the emotion file
        with open(file=download_result.emotion_file_path, mode="r") as f:
            emotion_reader = csv.reader(f, delimiter="\t")
            for text, labels in emotion_reader:
                parsed_labels = labels.split(sep=",")

                mapped_labels = list(
                    map(lambda x: self.emotion_map[x.strip()], parsed_labels)
                )

                emotion_labels = self.get_default_emotion_labels()

                emotion_labels.update({emotion: True for emotion in mapped_labels})

                records.append({"text": text, **emotion_labels})

        # Process the neutral file
        with open(file=download_result.neutral_file_path, mode="r") as f:
            neutral_reader = csv.reader(f, delimiter="\t")
            for line in neutral_reader:
                # Skip some malformed lines
                if len(line) != 2:
                    continue
                else:
                    label, text = line

                emotion_labels = self.get_default_emotion_labels()

                emotion_labels.update({"neutral": True})

                records.append({"text": text, **emotion_labels})

        # Convert to HF dataset
        hf_dataset = datasets.Dataset.from_list(
            mapping=records,
            info=datasets.DatasetInfo(
                dataset_name=self.name,
                description=self.metadata.description,
                citation=self.metadata.citation,
                homepage=self.metadata.homepage,
                license=self.metadata.license,
            ),
        )

        logger.info(f"Processing - Saving HuggingFace dataset: {data_subdir}")

        hf_dataset.save_to_disk(
            dataset_path=str(data_subdir),
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            num_proc=num_proc,
            storage_options=storage_options,
        )

        logger.info("Processing - Finished saving HuggingFace dataset.")

        shuffled_dataset = hf_dataset.shuffle()

        logger.info(
            f"Processing - Dataset sample: {json.dumps(obj=shuffled_dataset[0], sort_keys=True, indent=2)}"
        )

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

        logger.info("Processing - Finished dataset processing.")

        processing_result = XEDProcessingResult(data_subdir=data_subdir)

        return processing_result

    def teardown(
        self,
        download_result: XEDDownloadResult,
        processing_result: XEDProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
