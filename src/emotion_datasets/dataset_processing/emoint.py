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


EMOINT_METADATA = DatasetMetadata(
    description="The WASSA-2017 Shared Task on Emotion Intensity (EmoInt) dataset, as processed using 'emotion_datasets'. Unlike other emotion datasets, texts in this dataset are annotated not just for the dominant emotion, but for their intensity as well.",
    citation=(
        "@article{emotion_dataset_emoint,"
        "\n  title={WASSA-2017 shared task on emotion intensity},"
        "\n  author={Mohammad, Saif M and Bravo-Marquez, Felipe},"
        "\n  journal={arXiv preprint arXiv:1708.03700},"
        "\n  year={2017}"
        "\n}"
    ),
    homepage="http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html",
    license="",
    emotions=[
        "anger",
        "fear",
        "joy",
        "sadness",
    ],
    multilabel=True,
    continuous=True,
    system="Subset of common emotions anotated using best-worst scaling",
    domain="Twitter posts",
)


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


@dataclasses.dataclass
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

    metadata: typing.ClassVar[DatasetMetadata] = EMOINT_METADATA

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
                        ARRAY_AGG(id) AS ids,
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
