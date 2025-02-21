import shutil
import tempfile
import typing
import dataclasses
import os
import pathlib
import logging
import re

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
    update_samples,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CROWDFLOWER_METADATA = DatasetMetadata(
    description="The Emotion in Text dataset by CrowdFlower, as processed using 'emotion_datasets'. A dataset of tweets labelled into 1 of 13 different emotion classes. The original dataset is no longer publicly available, so this was taken from a secondary source.",
    citation="",
    homepage="",
    license="",
    emotions=[
        "anger",
        "boredom",
        "empty",
        "enthusiasm",
        "fun",
        "happiness",
        "hate",
        "love",
        "neutral",
        "relief",
        "sadness",
        "surprise",
        "worry",
    ],
    multilabel=False,
    continuous=False,
    system="Hashtags in twitter posts",
    domain="Twitter posts",
)


@dataclasses.dataclass
class CrowdFlowerDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    data_file_path: pathlib.Path


@dataclasses.dataclass
class CrowdFlowerProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path
    temp_dir: pathlib.Path


CORPUS_MATCHER = re.compile(r"<instance id=\"(.*?)\">(.*?)</instance>")


@dataclasses.dataclass
class CrowdFlowerProcessor(DatasetBase):
    name: str = "CrowdFlower"

    url: str = "https://raw.githubusercontent.com/tlkh/text-emotion-classification/refs/heads/master/dataset/original/text_emotion.csv"

    metadata: typing.ClassVar[DatasetMetadata] = CROWDFLOWER_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> CrowdFlowerDownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        file_name = pathlib.Path(self.url).name

        logger.info(f"Download - Downloading data file: {file_name}")

        downloaded_file_path = downloads_dir / "data.csv"

        try:
            download(
                url=self.url,
                file_path=downloaded_file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download data file ({file_name}). Encountered the following exception: {e}"
            )

        download_result = CrowdFlowerDownloadResult(
            downloads_subdir=downloads_subdir, data_file_path=downloaded_file_path
        )

        return download_result

    def process_files(
        self,
        download_result: CrowdFlowerDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> CrowdFlowerProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_data_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_data_dir}")

            temp_db = duckdb.connect(str(temp_data_dir / "metadata.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_data_dir / 'metadata.db'}"
            )

            logger.info(msg="Processing - Ingesting data file using duckdb")

            temp_db.sql(
                f"""
                CREATE TABLE temp
                AS
                    SELECT
                        tweet_id,
                        sentiment,
                        content AS text
                    FROM read_csv('{str(download_result.data_file_path)}',
                        header = true,
                        columns = {{
                            'tweet_id': 'VARCHAR',
                            'sentiment': 'VARCHAR',
                            'author': 'VARCHAR',
                            'content': 'VARCHAR',
                        }})
                """
            )

            # Pivot on the sentiment column
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS
                    SELECT
                        tweet_id,
                        text,
                        IFNULL(anger, false) AS anger,
                        IFNULL(boredom, false) AS boredom,
                        IFNULL(empty, false) AS empty,
                        IFNULL(enthusiasm, false) AS enthusiasm,
                        IFNULL(fun, false) AS fun,
                        IFNULL(happiness, false) AS happiness,
                        IFNULL(hate, false) AS hate,
                        IFNULL(love, false) AS love,
                        IFNULL(neutral, false) AS neutral,
                        IFNULL(relief, false) AS relief,
                        IFNULL(sadness, false) AS sadness,
                        IFNULL(surprise, false) AS surprise,
                        IFNULL(worry, false) AS worry
                    FROM (
                        PIVOT temp
                        ON sentiment
                    )
                """
            )

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

        update_samples(
            data_subdir=data_subdir,
            dataset_name=self.name,
            storage_options=storage_options,
        )

        logger.info("Processing - Finished dataset processing.")

        processing_result = CrowdFlowerProcessingResult(
            temp_dir=temp_data_dir, data_subdir=data_subdir
        )

        return processing_result

    def teardown(
        self,
        download_result: CrowdFlowerDownloadResult,
        processing_result: CrowdFlowerProcessingResult,
        storage_options: dict,
    ) -> None:
        download_result.data_file_path.unlink()

        shutil.rmtree(path=download_result.downloads_subdir)

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
