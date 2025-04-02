import shutil
import tempfile
import typing
import dataclasses
import os
import pathlib
import logging

import datasets
import gdown
import duckdb

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadResult,
    ProcessingResult,
    DownloadError,
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

WASSA22_METADATA = DatasetMetadata(
    description="The WASSA 2022 Shared Task dataset, as processed by 'emotion_datasets'. This task aims at developing models which can predict empathy and emotion based on essays written in reaction to news articles where there is harm to a person, group, or other.",
    citation=(
        "@inproceedings{emotion_dataset_wassa2022,"
        "\n  title={WASSA 2022 shared task: Predicting empathy, emotion and personality in reaction to news stories},"
        "\n  author={Barriere, Valentin and Tafreshi, Shabnam and Sedoc, Jo{\\~a}o and Alqahtani, Sawsan},"
        "\n  booktitle={Proceedings of the 12th Workshop on Computational Approaches to Subjectivity, Sentiment \\& Social Media Analysis},"
        "\n  pages={214--227},"
        "\n  year={2022}"
        "\n}"
    ),
    homepage="https://codalab.lisn.upsaclay.fr/competitions/834#learn_the_details-overview",
    license="",
    emotions=[
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutral",
        "sadness",
        "surprise",
        "empathy",
        "distress",
    ],
    multilabel=False,
    continuous=False,
    system="Ekman basic emotions, along with continuous scores for empathy and distress",
    domain="Essays",
)


@dataclasses.dataclass
class WASSA22DownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    train_file_path: pathlib.Path
    dev_features_file_path: pathlib.Path
    dev_labels_file_path: pathlib.Path


@dataclasses.dataclass
class WASSA22ProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_subdir: pathlib.Path


@dataclasses.dataclass
class WASSA22Processor(DatasetBase):
    name: str = "WASSA22"

    train_url: str = (
        "https://drive.google.com/file/d/19U-iFap2gJPTSR8pq52jF276CM22wex9/view"
    )

    dev_features_url: str = (
        "https://drive.google.com/file/d/1Behbb9DWcVozRf1diuT4-Nfu2GeBRz8f/view"
    )

    dev_labels_url: str = (
        "https://drive.google.com/file/d/1BSN3YeEwSzN8yOjoIYLA-9Atol986ub5/view"
    )

    metadata: typing.ClassVar[DatasetMetadata] = WASSA22_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> WASSA22DownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        # Train file
        logger.info(f"Download - Downloading train file: {self.train_url}")
        train_file_path = downloads_subdir / "train.csv"
        try:
            gdown.download(
                url=self.train_url,
                output=str(train_file_path),
                fuzzy=True,
            )

        except Exception as e:
            raise DownloadError(
                f"Could not download train file. Encountered the following exception: {e}"
            )

        # Dev features file
        logger.info(
            f"Download - Downloading dev features file: {self.dev_features_url}"
        )
        dev_features_file_path = downloads_subdir / "dev_features.csv"
        try:
            gdown.download(
                url=self.dev_features_url,
                output=str(dev_features_file_path),
                fuzzy=True,
            )

        except Exception as e:
            raise DownloadError(
                f"Could not download dev features file. Encountered the following exception: {e}"
            )

        # Dev labels file
        logger.info(f"Download - Downloading dev labels file: {self.dev_labels_url}")
        dev_labels_file_path = downloads_subdir / "dev_labels.csv"
        try:
            gdown.download(
                url=self.dev_labels_url,
                output=str(dev_labels_file_path),
                fuzzy=True,
            )

        except Exception as e:
            raise DownloadError(
                f"Could not download dev labels file. Encountered the following exception: {e}"
            )

        download_result = WASSA22DownloadResult(
            downloads_subdir=downloads_subdir,
            train_file_path=train_file_path,
            dev_features_file_path=dev_features_file_path,
            dev_labels_file_path=dev_labels_file_path,
        )

        return download_result

    def process_files(
        self,
        download_result: WASSA22DownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> WASSA22ProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_data_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_data_dir}")

            temp_db = duckdb.connect(str(temp_data_dir / "temp.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_data_dir / 'temp.db'}"
            )

            logger.info(msg="Processing - Merging files using duckdb")

            # Create a table
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp (
                    id VARCHAR,
                    article_id VARCHAR,
                    text VARCHAR,
                    emotion VARCHAR,
                    empathy FLOAT,
                    distress FLOAT,
                )
                """
            )

            # Read the train file into the table
            temp_db.sql(
                f"""
                INSERT INTO temp
                    SELECT
                        message_id AS id,
                        article_id AS article_id,
                        essay AS text,
                        emotion,
                        empathy,
                        distress,
                    FROM read_csv('{str(download_result.train_file_path)}',
                        header = true
                    )
                """
            )

            # Read the dev files into the table
            # Join by row (i.e., concatenate)
            temp_db.sql(
                f"""
                INSERT INTO temp
                    SELECT *
                    FROM (
                        SELECT
                            message_id AS id,
                            article_id AS article_id,
                            essay AS text
                        FROM read_csv('{str(download_result.dev_features_file_path)}',
                            header = true
                        )
                    ) POSITIONAL JOIN
                    (
                        SELECT
                            column02 AS emotion,
                            column00 AS empathy,
                            column01 AS distress
                        FROM read_csv('{str(download_result.dev_labels_file_path)}',
                            header = false
                        )
                    )
                """
            )

            # Pivot on the emotion column
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE TEMP
                AS (
                    SELECT
                        id,
                        article_id,
                        text,
                        empathy,
                        distress,
                        IFNULL(anger, false) AS anger,
                        IFNULL(disgust, false) AS disgust,
                        IFNULL(fear, false) AS fear,
                        IFNULL(joy, false) AS joy,
                        IFNULL(neutral, false) AS neutral,
                        IFNULL(sadness, false) AS sadness,
                        IFNULL(surprise, false) AS surprise
                    FROM (
                        PIVOT temp
                        ON emotion
                        USING BOOL_AND(true)
                    )
                )
                """
            )

            logger.info("Processing - Merged files")

            handoff_file = temp_data_dir / "output.parquet"

            temp_db.sql(f"COPY temp TO '{str(handoff_file)}' (FORMAT PARQUET);")

            logger.debug(f"Processing - Wrote to handoff file: {handoff_file}")

            hf_dataset: datasets.Dataset = datasets.Dataset.from_parquet(
                path_or_paths=str(handoff_file),
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

        processing_result = WASSA22ProcessingResult(
            temp_dir=temp_data_dir,
            data_subdir=data_subdir,
        )

        return processing_result

    def teardown(
        self,
        download_result: WASSA22DownloadResult,
        processing_result: WASSA22ProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(path=processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
