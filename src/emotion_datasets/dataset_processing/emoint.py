import json
import shutil
import tempfile
import typing
import dataclasses
import os
import pathlib
import logging

import datasets
import duckdb

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadResult,
    ProcessingResult,
    DownloadError,
)
from emotion_datasets.utils import download, get_file_stats, update_manifest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclasses.dataclass
class EmoIntDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    downloaded_files: typing.List[pathlib.Path]


@dataclasses.dataclass
class EmoIntProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_subdir: pathlib.Path


@dataclasses.dataclass(kw_only=True, frozen=True)
class EmoIntProcessor(DatasetBase):
    name: str = "EmoInt"

    urls: typing.List[str] = dataclasses.field(
        default_factory=lambda: [
            "http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/anger-ratings-0to1.train.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/fear-ratings-0to1.train.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/joy-ratings-0to1.train.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Train%20Data/sadness-ratings-0to1.train.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/anger-ratings-0to1.dev.gold.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/fear-ratings-0to1.dev.gold.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/joy-ratings-0to1.dev.gold.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Dev%20Data%20With%20Gold/sadness-ratings-0to1.dev.gold.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/anger-ratings-0to1.test.gold.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/fear-ratings-0to1.test.gold.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/joy-ratings-0to1.test.gold.txt",
            "http://saifmohammad.com/WebDocs/EmoInt%20Test%20Gold%20Data/sadness-ratings-0to1.test.gold.txt",
        ]
    )

    def download_files(self, downloads_dir: pathlib.Path) -> EmoIntDownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        downloaded_files = []
        for i, url in enumerate(self.urls):
            downloaded_file_name = pathlib.Path(url).name

            logger.info(
                f"Download - Downloading file {i}/{len(self.urls)}: {downloaded_file_name}"
            )

            emotion = downloaded_file_name.split("-")[0]
            split = downloaded_file_name.split(".")[1]

            downloaded_file_path = downloads_subdir / (emotion + "-" + split + ".csv")

            try:
                download(
                    url=url,
                    file_path=downloaded_file_path,
                )
            except Exception as e:
                raise DownloadError(
                    f"Could not download file ({downloaded_file_name}). Encountered the following exception: {e}"
                )

            downloaded_files.append(downloaded_file_path)

        download_result = EmoIntDownloadResult(
            downloads_subdir=downloads_subdir, downloaded_files=downloaded_files
        )

        return download_result

    def process_files(
        self,
        download_result: EmoIntDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> EmoIntProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        downloaded_files_pattern = str(
            object=download_result.downloads_subdir / "*.csv"
        )

        logger.debug(f"Processing - Looking for files in {downloaded_files_pattern}")

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_dir}")

            temp_db = duckdb.connect(str(temp_dir / "temp.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_dir / 'temp.db'}"
            )

            logger.info(msg="Processing - Merging files using duckdb")

            # First load in the separate csv files
            # Each text is only annotated for a single emotion
            temp_db.sql(
                f"""
                CREATE TABLE temp
                AS (
                    SELECT *
                    FROM read_csv('{downloaded_files_pattern}',
                        header = false,
                        columns = {{
                            "id": "INTEGER",
                            "text": "VARCHAR",
                            "emotion": "VARCHAR",
                            "intensity": "FLOAT"
                        }}
                    )
                )
                """
            )

            # Then pivot so that each row is annotated for each emotion
            # Introduces many NULL values though
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS (
                    PIVOT temp
                    ON emotion
                    USING
                        sum(intensity)
                )
                """
            )

            # Finally, merge any duplicates together
            # Now each row can have multiple emotions
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        ARRAY_AGG(id) as ids,
                        text,
                        MEAN(anger) AS anger,
                        MEAN(fear) AS fear,
                        MEAN(joy) AS joy,
                        MEAN(sadness) AS sadness,
                    FROM temp
                    GROUP BY text
                )
                """
            )

            logger.info("Processing - Merged files")

            handoff_file = temp_dir / "output.parquet"

            temp_db.sql(f"COPY temp TO '{str(handoff_file)}' (FORMAT PARQUET);")

            logger.debug(f"Processing - Wrote to handoff file: {handoff_file}")

            hf_dataset: datasets.Dataset = datasets.Dataset.from_parquet(
                str(handoff_file),
                cache_dir=str(temp_dir),
                keep_in_memory=False,
            )  # type: ignore

            hf_dataset.info.description = "The WASSA-2017 Shared Task on Emotion Intensity (EmoInt) dataset, as processed using 'emotion_datasets'. Unlike other emotion datasets, texts in this dataset are annotated not just for the dominant emotion, but for their intensity as well."

            hf_dataset.info.citation = (
                "@article{mohammad2017wassa,"
                "  title={WASSA-2017 shared task on emotion intensity},"
                "  author={Mohammad, Saif M and Bravo-Marquez, Felipe},"
                "  journal={arXiv preprint arXiv:1708.03700},"
                "  year={2017}"
                "}"
            )

            hf_dataset.info.homepage = (
                "http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html"
            )

            hf_dataset.info.license = ""

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

        logger.info("Processing - Dataset sample: ")

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

        logger.info("Processing - Finished dataset processing.")

        processing_result = EmoIntProcessingResult(
            temp_dir=temp_dir, data_subdir=data_subdir
        )

        return processing_result

    def teardown(
        self,
        download_result: EmoIntDownloadResult,
        processing_result: EmoIntProcessingResult,
        storage_options: dict,
    ) -> None:
        for file_path in download_result.downloaded_files:
            file_path.unlink()

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
