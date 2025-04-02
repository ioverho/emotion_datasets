import shutil
import tempfile
import typing
import dataclasses
import os
import pathlib
import logging
import zipfile

import datasets
import duckdb
import gdown

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

CANCEREMO_METADATA = DatasetMetadata(
    description="The CancerEmo dataset, as processed by 'emotion_datasets'. CancerEmo is an emotion dataset created from an online health community and annotated with eight fine-grained emotions.",
    citation=(
        "@inproceedings{emotion_dataset_canceremo,"
        "\n    title = '{C}ancer{E}mo: A Dataset for Fine-Grained Emotion Detection',"
        "\n    author = 'Sosea, Tiberiu  and"
        "\n      Caragea, Cornelia',"
        "\n    booktitle = 'Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)',"
        "\n    month = nov,"
        "\n    year = '2020',"
        "\n    address = 'Online',"
        "\n    publisher = 'Association for Computational Linguistics',"
        "\n    url = 'https://www.aclweb.org/anthology/2020.emnlp-main.715',"
        "\n    doi = '10.18653/v1/2020.emnlp-main.715',"
        "\n    pages = '8892--8904',"
        "\n}"
    ),
    homepage="https://github.com/tsosea2/CancerEmo?tab=readme-ov-file",
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
    ],
    multilabel=True,
    continuous=False,
    system="Plutchik-8 emotions",
    domain="Cancer survivors internet forum",
)


@dataclasses.dataclass
class CancerEmoDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path


@dataclasses.dataclass
class CancerEmoProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_subdir: pathlib.Path


@dataclasses.dataclass
class CancerEmoProcessor(DatasetBase):
    name: str = "CancerEmo"

    url: str = "https://drive.google.com/file/d/1kYJaCYBL1B8dCJYojp34W2C6smUfYPvv/view"

    metadata: typing.ClassVar[DatasetMetadata] = CANCEREMO_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> CancerEmoDownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        zip_filepath = downloads_subdir / "dataset.zip"

        try:
            gdown.download(url=self.url, output=str(zip_filepath), fuzzy=True)
        except Exception as e:
            raise DownloadError(
                f"Could not download CancerEmo dataset from '{self.url}'. Encountered the following exception: {e}"
            )

        logger.info(f"Download - Downloaded CancerEmo dataset: {zip_filepath}")

        with zipfile.ZipFile(zip_filepath, "r") as f:
            f.extractall(downloads_subdir)

        download_result = CancerEmoDownloadResult(
            downloads_subdir=downloads_subdir,
        )

        return download_result

    def process_files(
        self,
        download_result: CancerEmoDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> CancerEmoProcessingResult:
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

            # Create an initial table
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp (
                    text VARCHAR,
                    emotion VARCHAR,
                    label BOOL,
                )
                """
            )

            # Read in every emoption file and unpivot
            # Columns: text, emotion, label
            for emotion in self.metadata.emotions:
                title_case_emotion = emotion[0].upper() + emotion[1:]

                csv_file_path = str(
                    download_result.downloads_subdir / f"{title_case_emotion}_anon.csv"
                )

                # Read the train file into the table
                temp_db.sql(
                    f"""
                    INSERT INTO temp
                        UNPIVOT (
                            SELECT
                                Sentence as text,
                                {title_case_emotion} as {emotion},
                            FROM read_csv('{csv_file_path}',
                                header = true
                            )
                        )
                        ON {emotion}
                        INTO
                            NAME emotion
                            VALUE label
                    """
                )

            # Pivot the aggregated emotion table
            # Columns: text, emotion 1, emotion 2, ..., emotion K
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS
                    PIVOT temp
                    ON emotion
                    USING ANY_VALUE(label)
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
            dataset_metadata=self.metadata,
        )

        update_samples(
            data_subdir=data_subdir,
            dataset_name=self.name,
            storage_options=storage_options,
        )

        logger.info("Processing - Finished dataset processing.")

        processing_result = CancerEmoProcessingResult(
            data_subdir=data_subdir, temp_dir=temp_dir
        )

        return processing_result

    def teardown(
        self,
        download_result: CancerEmoDownloadResult,
        processing_result: CancerEmoProcessingResult,
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
