from collections import Counter
import json
import shutil
import typing
import dataclasses
import os
import pathlib
import logging

import datasets

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadResult,
    ProcessingResult,
    DownloadError,
    DatasetMetadata,
)
from emotion_datasets.utils import (
    download,
    get_file_stats,
    update_manifest,
    update_bib_file,
    update_samples,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HURRICANES24_METADATA = DatasetMetadata(
    description="The Hurricanes dataset, as processed by 'emotion_datasets'. The datasets include tweets for 3 different hurricanes: Harvey, Irma and Maria. Each tweet has MTurk annotations for the fine-grained Plutchik-24 emotion system.",
    citation=(
        "@inproceedings{emotion_datasets_hurricanes,"
        "\n   author={Desai, Shrey and Caragea, Cornelia and Li, Junyi Jessy},"
        "\n   title={{Detecting Perceived Emotions in Hurricane Disasters}},"
        "\n   booktitle={Proceedings of the Association for Computational Linguistics (ACL)},"
        "\n   year={2020},"
        "\n}"
    ),
    homepage="https://github.com/shreydesai/hurricane/tree/master",
    license="",
    emotions=[
        "admiration",
        "ecstasy",
        "interest",
        "sadness",
        "vigilance",
        "acceptance",
        "amazement",
        "anger",
        "annoyance",
        "anticipation",
        "apprehension",
        "boredom",
        "disgust",
        "distraction",
        "fear",
        "grief",
        "joy",
        "loathing",
        "pensiveness",
        "rage",
        "serenity",
        "surprise",
        "terror",
        "trust",
    ],
    multilabel=True,
    continuous=False,
    system="Plutchik-24 emotions",
    domain="Twitter posts about hurricanes",
)


@dataclasses.dataclass
class Hurricanes24DownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    downloaded_file_paths: typing.List[pathlib.Path]


@dataclasses.dataclass
class Hurricanes24ProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path


@dataclasses.dataclass
class Hurricanes24Processor(DatasetBase):
    name: str = "Hurricanes24"

    urls: typing.List[str] = dataclasses.field(
        default_factory=lambda: [
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_raw/harvey.jsonl",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_raw/irma.jsonl",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_raw/maria.jsonl",
        ]
    )

    metadata: typing.ClassVar[DatasetMetadata] = HURRICANES24_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> Hurricanes24DownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        downloaded_file_paths = []
        for i, url in enumerate(self.urls):
            file_name = pathlib.Path(url).name
            file_path = downloads_subdir / file_name

            logger.info(
                f"Download - Downloading hurricane file {i}/{len(self.urls)}: {file_name}"
            )

            try:
                download(
                    url=url,
                    file_path=file_path,
                )
            except Exception as e:
                raise DownloadError(
                    f"Could not download hurricane file ({file_name}). Encountered the following exception: {e}"
                )

            downloaded_file_paths.append(file_path)

        download_result = Hurricanes24DownloadResult(
            downloads_subdir=downloads_subdir,
            downloaded_file_paths=downloaded_file_paths,
        )

        return download_result

    def process_files(
        self,
        download_result: Hurricanes24DownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> Hurricanes24ProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        # Process the emotion annotation files
        records = []
        for file_path in download_result.downloaded_file_paths:
            hurricane = file_path.stem

            with open(file_path, mode="r") as f:
                for line in f:
                    line_data = json.loads(line.strip())

                    emotion_votes = Counter()
                    for _, annotation in line_data["annotations"].items():
                        for emotion, presence in annotation.items():
                            emotion_votes[emotion] += 1 if presence else 0

                    record = {
                        "hurricane": hurricane,
                        "text": line_data["text"],
                        **dict(emotion_votes),
                    }

                    records.append(record)

            logger.info(
                f"Processing - Finished processing hurricane {hurricane}. Current number of records: {len(records)}"
            )

        # Hand off to HuggingFace
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
            dataset_metadata=HURRICANES24_METADATA,
        )

        update_samples(
            data_subdir=data_subdir,
            dataset_name=self.name,
            storage_options=storage_options,
        )

        logger.info("Processing - Finished dataset processing.")

        processing_result = Hurricanes24ProcessingResult(data_subdir=data_subdir)

        return processing_result

    def teardown(
        self,
        download_result: Hurricanes24DownloadResult,
        processing_result: Hurricanes24ProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
