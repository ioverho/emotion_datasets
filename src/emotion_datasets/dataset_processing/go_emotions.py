import typing
import dataclasses
import os
import pathlib
import logging
import tempfile
import shutil

import duckdb
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


GOEMOTIONS_METADATA = DatasetMetadata(
    description="The GoEmotions dataset, as processed using 'emotion_datasets'. GoEmotions is a corpus of 58k carefully curated comments extracted from Reddit, with human annotations to 27 emotion categories or Neutral.",
    citation=(
        "@inproceedings{emotion_dataset_go_emotions,"
        "\n   author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},"
        "\n   booktitle = {58th Annual Meeting of the Association for Computational Linguistics (ACL)},"
        "\n   title = {{GoEmotions: A Dataset of Fine-Grained Emotions}},"
        "\n   year = {2020}"
        "\n}"
    ),
    homepage="https://github.com/google-research/google-research/tree/master/goemotions",
    license="",
    emotions=[
        "admiration",
        "amusement",
        "anger",
        "annoyance",
        "approval",
        "caring",
        "confusion",
        "curiosity",
        "desire",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "excitement",
        "fear",
        "gratitude",
        "grief",
        "joy",
        "love",
        "nervousness",
        "optimism",
        "pride",
        "realization",
        "relief",
        "remorse",
        "sadness",
        "surprise",
        "neutral",
    ],
    multilabel=True,
    continuous=False,
    system="Custom hierarchical emotion system",
    domain="Reddit posts",
)


@dataclasses.dataclass
class GoEmotionsDownloadResult(DownloadResult):
    metadata_files: typing.List[pathlib.Path]
    data_files: typing.List[pathlib.Path]


@dataclasses.dataclass
class GoEmotionsProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class GoEmotionsProcessor(DatasetBase):
    name: str = "GoEmotions"

    files: typing.List[str] = dataclasses.field(
        default_factory=lambda: [
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv",
        ]
    )

    metadata_files: typing.List[str] = dataclasses.field(
        default_factory=lambda: [
            "https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/goemotions/data/emotions.txt",
            "https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/goemotions/data/sentiment_dict.json",
            "https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/goemotions/data/ekman_mapping.json",
        ]
    )

    metadata: typing.ClassVar[DatasetMetadata] = GOEMOTIONS_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> GoEmotionsDownloadResult:
        os.makedirs(downloads_dir / self.name, exist_ok=True)

        # Metadata files
        downloaded_metadata_files = []
        for i, metadata_file in enumerate(self.metadata_files):
            metadata_file_name = pathlib.Path(metadata_file).name

            logger.info(
                f"Download - Downloading metadata file {i + 1}/{len(self.metadata_files)}: {metadata_file_name}"
            )
            logger.info(f"Download - Source: {metadata_file}")

            metadata_file_path = downloads_dir / self.name / metadata_file_name

            try:
                download(
                    url=metadata_file,
                    file_path=metadata_file_path,
                )
            except Exception as e:
                raise DownloadError(
                    f"Could not download metadata file ({metadata_file}). Raises the following exception: {e}"
                )

            downloaded_metadata_files.append(metadata_file_path)

        # Data files
        downloaded_data_files = []
        for i, data_file in enumerate(self.files):
            data_file_name = pathlib.Path(data_file).name

            logger.info(
                f"Download - Downloading data file {i + 1}/{len(self.files)}: {data_file}"
            )
            logger.info(f"Download - Source: {data_file}")

            data_file_path = downloads_dir / self.name / data_file_name

            try:
                download(
                    url=data_file,
                    file_path=data_file_path,
                )
            except Exception as e:
                raise DownloadError(
                    f"Could not download data file ({data_file}). Raises the following exception: {e}"
                )

            downloaded_data_files.append(data_file_path)

        download_result = GoEmotionsDownloadResult(
            metadata_files=downloaded_metadata_files, data_files=downloaded_data_files
        )

        return download_result

    def process_files(
        self,
        download_result: GoEmotionsDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> GoEmotionsProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        downloaded_files_pattern = str(
            download_result.data_files[0].with_stem(
                download_result.data_files[0].stem[:-1] + "*"
            )
        )

        logger.debug(
            msg=f"Processing - Looking for files in {downloaded_files_pattern}"
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_data_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_data_dir}")

            temp_db = duckdb.connect(str(temp_data_dir / "temp.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_data_dir / 'temp.db'}"
            )

            logger.info(msg="Processing - Merging files using duckdb")

            temp_db.sql(
                f"""
                CREATE TABLE temp
                AS (
                    SELECT *
                    FROM read_csv('{downloaded_files_pattern}',
                    header = true,
                    columns = {{
                        'text': 'VARCHAR',
                        'id': 'VARCHAR',
                        'author': 'VARCHAR',
                        'subreddit': 'VARCHAR',
                        'link_id': 'VARCHAR',
                        'parent_id': 'VARCHAR',
                        'created_utc': 'FLOAT',
                        'rater_id': 'UINTEGER',
                        'example_very_unclear': 'BOOL',
                        'admiration': 'BOOL',
                        'amusement': 'BOOL',
                        'anger': 'BOOL',
                        'annoyance': 'BOOL',
                        'approval': 'BOOL',
                        'caring': 'BOOL',
                        'confusion': 'BOOL',
                        'curiosity': 'BOOL',
                        'desire': 'BOOL',
                        'disappointment': 'BOOL',
                        'disapproval': 'BOOL',
                        'disgust': 'BOOL',
                        'embarrassment': 'BOOL',
                        'excitement': 'BOOL',
                        'fear': 'BOOL',
                        'gratitude': 'BOOL',
                        'grief': 'BOOL',
                        'joy': 'BOOL',
                        'love': 'BOOL',
                        'nervousness': 'BOOL',
                        'optimism': 'BOOL',
                        'pride': 'BOOL',
                        'realization': 'BOOL',
                        'relief': 'BOOL',
                        'remorse': 'BOOL',
                        'sadness': 'BOOL',
                        'surprise': 'BOOL',
                        'neutral': 'BOOL'
                    }})
                )
            """
            )

            logger.info("Processing - Merged files")

            handoff_file = temp_data_dir / "output.parquet"

            temp_db.sql(f"COPY temp TO '{str(handoff_file)}' (FORMAT PARQUET);")

            logger.debug(f"Processing - Wrote to handoff file: {handoff_file}")

            hf_dataset: datasets.Dataset = datasets.Dataset.from_parquet(
                str(handoff_file),
                cache_dir=str(temp_data_dir),
                keep_in_memory=False,
            )  # type: ignore

            logger.info("Processing - Ingested handoff file using HuggingFace")

            hf_dataset.info.dataset_name = self.name
            hf_dataset.info.description = self.metadata.description
            hf_dataset.info.citation = self.metadata.citation
            hf_dataset.info.homepage = self.metadata.homepage
            hf_dataset.info.license = self.metadata.license

            logger.info(
                f"Processing - HuggingFace dataset has {hf_dataset.num_rows} rows"
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
            "metadata": [],
        }
        for fp in data_subdir.glob("*"):
            file_stats = get_file_stats(fp=fp, data_dir=data_dir)

            data_dir_summary["data"].append(file_stats)

        logger.info("Processing - Moving metadata files.")

        for metadata_file_path in download_result.metadata_files:
            shutil.move(
                src=metadata_file_path,
                dst=data_subdir / metadata_file_path.name,
            )

            file_stats = get_file_stats(
                fp=data_subdir / metadata_file_path.name, data_dir=data_dir
            )

            data_dir_summary["metadata"].append(file_stats)

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

        processing_result = GoEmotionsProcessingResult(
            temp_dir=temp_data_dir, data_dir=data_subdir
        )

        return processing_result

    def teardown(
        self,
        download_result: GoEmotionsDownloadResult,
        processing_result: GoEmotionsProcessingResult,
        storage_options: dict,
    ) -> None:
        for data_file_path in download_result.data_files:
            data_file_path.unlink()

        for metadata_file_path in download_result.metadata_files:
            if metadata_file_path.exists():
                logger.debug(
                    f"Teardown - Metadata file '{metadata_file_path}' still exists..."
                )
                metadata_file_path.unlink()

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(path=processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_dir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
