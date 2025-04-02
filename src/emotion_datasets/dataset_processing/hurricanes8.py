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

HURRICANES8_METADATA = DatasetMetadata(
    description="The Hurricanes dataset, as processed by 'emotion_datasets'. The datasets include tweets for 3 different hurricanes: Harvey, Irma and Maria. Each tweet has MTurk annotations for the fine-grained Plutchik-24 emotion system.",
    citation=(
        "@inproceedings{emotion_dataset_hurricanes,"
        "\n   author={Desai, Shrey and Caragea, Cornelia and Li, Junyi Jessy},"
        "\n   title={{Detecting Perceived Emotions in Hurricane Disasters}},"
        "\n   booktitle={Proceedings of the Association for Computational Linguistics (ACL)},"
        "\n   year={2020},"
        "\n}"
    ),
    homepage="https://github.com/shreydesai/hurricane/tree/master",
    license="",
    emotions=[
        "aggressiveness",
        "awe",
        "contempt",
        "disapproval",
        "love",
        "optimism",
        "remorse",
        "submission",
    ],
    multilabel=True,
    continuous=False,
    system="Plutchik-8 emotions",
    domain="Twitter posts about hurricanes",
)


@dataclasses.dataclass
class Hurricanes8DownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    downloaded_file_paths: typing.List[pathlib.Path]


@dataclasses.dataclass
class Hurricanes8ProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_subdir: pathlib.Path


@dataclasses.dataclass
class Hurricanes8Processor(DatasetBase):
    name: str = "Hurricanes8"

    urls: typing.List[str] = dataclasses.field(
        default_factory=lambda: [
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//aggressiveness_train.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//aggressiveness_valid.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//aggressiveness_test.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//awe_train.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//awe_valid.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//awe_test.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//contempt_train.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//contempt_valid.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//contempt_test.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//disapproval_train.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//disapproval_valid.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//disapproval_test.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//love_train.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//love_valid.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//love_test.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//optimism_train.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//optimism_valid.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//optimism_test.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//remorse_train.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//remorse_valid.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//remorse_test.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//submission_train.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//submission_valid.csv",
            "https://raw.githubusercontent.com/shreydesai/hurricane/refs/heads/master/datasets_binary//submission_test.csv",
        ]
    )

    metadata: typing.ClassVar[DatasetMetadata] = HURRICANES8_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> Hurricanes8DownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        downloaded_file_paths = []
        for i, url in enumerate(self.urls):
            file_name = pathlib.Path(url).name
            file_path = downloads_subdir / file_name

            logger.info(
                f"Download - Downloading binary emotion file {i}/{len(self.urls)}: {file_name}"
            )

            try:
                download(
                    url=url,
                    file_path=file_path,
                )
            except Exception as e:
                raise DownloadError(
                    f"Could not download binary emotion file ({file_name}). Encountered the following exception: {e}"
                )

            downloaded_file_paths.append(file_path)

        download_result = Hurricanes8DownloadResult(
            downloads_subdir=downloads_subdir,
            downloaded_file_paths=downloaded_file_paths,
        )

        return download_result

    def process_files(
        self,
        download_result: Hurricanes8DownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> Hurricanes8ProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_dir}")

            temp_db = duckdb.connect(str(temp_dir / "temp.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_dir / 'temp.db'}"
            )

            logger.info(msg="Processing - Merging files using duckdb")

            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp (
                    text VARCHAR,
                    emotion VARCHAR,
                    value INTEGER,
                )
                """
            )

            # Ingest all binary emotion files
            # Should be one train, val, test file for each emotion
            for emotion in self.metadata.emotions:
                file_pattern = str(
                    download_result.downloads_subdir / (emotion + "*.csv")
                )

                temp_db.sql(
                    f"""
                    INSERT INTO temp
                        UNPIVOT (
                            SELECT *
                            FROM read_csv('{file_pattern}',
                                header = true
                            )
                        )
                        ON {emotion}
                        INTO
                            NAME emotion
                            VALUE value
                    """
                )

            # Unpivot the table and aggregate across files
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        text,
                        CAST(IFNULL(aggressiveness, FALSE) AS BOOL) AS aggressiveness,
                        CAST(IFNULL(awe, FALSE) AS BOOL) AS awe,
                        CAST(IFNULL(contempt, FALSE) AS BOOL) AS contempt,
                        CAST(IFNULL(disapproval, FALSE) AS BOOL) AS disapproval,
                        CAST(IFNULL(love, FALSE) AS BOOL) AS love,
                        CAST(IFNULL(optimism, FALSE) AS BOOL) AS optimism,
                        CAST(IFNULL(remorse, FALSE) AS BOOL) AS remorse,
                        CAST(IFNULL(submission, FALSE) AS BOOL) AS submission
                    FROM (
                        PIVOT temp
                        ON emotion
                        USING sum(value)
                    )
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
            dataset_metadata=HURRICANES8_METADATA,
        )

        update_samples(
            data_subdir=data_subdir,
            dataset_name=self.name,
            storage_options=storage_options,
        )

        logger.info("Processing - Finished dataset processing.")

        processing_result = Hurricanes8ProcessingResult(
            data_subdir=data_subdir, temp_dir=temp_dir
        )

        return processing_result

    def teardown(
        self,
        download_result: Hurricanes8DownloadResult,
        processing_result: Hurricanes8ProcessingResult,
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
