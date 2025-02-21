import dataclasses
import os
import pathlib
import logging
import tempfile
import typing
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

ISEAR_METADATA = DatasetMetadata(
    description="The International Survey On Emotion Antecedents And Reactions (ISEAR) dataset, as processed using 'emotion_datasets'. Over a period of many years during the 1990s, a large group of psychologists all over the world collected data in the ISEAR project. Student respondents, both psychologists and non-psychologists, were asked to report situations in which they had experienced all of 7 major emotions. The final data set thus contained reports on seven emotions each by close to 3000 respondents in 37 countries on all 5 continents.",
    citation=(
        "@article{emotion_dataset_isear,"
        "\n    title={Evidence for universality and cultural variation of differential emotion response patterning.},"
        "\n    author={Scherer, Klaus R and Wallbott, Harald G},"
        "\n    journal={Journal of personality and social psychology},"
        "\n    volume={66},"
        "\n    number={2},"
        "\n    pages={310},"
        "\n    year={1994},"
        "\n    publisher={American Psychological Association}"
        "\n}"
    ),
    homepage="https://www.unige.ch/cisa/research/materials-and-online-research/research-material/",
    license="CC BY-NC-SA 3.0",
    emotions=[
        "anger",
        "disgust",
        "fear",
        "guilt",
        "joy",
        "sadness",
        "shame",
    ],
    multilabel=False,
    continuous=False,
    system="Situations in which a subject experienced one of 7 major emotions",
    domain="Situation descriptions",
)


@dataclasses.dataclass
class ISEARDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    data_file_path: pathlib.Path


@dataclasses.dataclass
class ISEARProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class ISEARProcessor(DatasetBase):
    name: str = "ISEAR"

    data_file: str = "https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/refs/heads/master/isear.csv"

    metadata: typing.ClassVar[DatasetMetadata] = ISEAR_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> ISEARDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(downloads_subdir, exist_ok=True)

        # Download the data files
        data_file_name = pathlib.Path(self.data_file).name

        logger.info(f"Download - Downloading data file: {data_file_name}")
        logger.info(f"Download - Source: {self.data_file}")

        data_file_path = downloads_dir / self.name / data_file_name

        try:
            download(
                url=self.data_file,
                file_path=data_file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download EmoBank data file ({self.data_file}). Raises the following exception: {e}"
            )

        download_result = ISEARDownloadResult(
            downloads_subdir=downloads_subdir, data_file_path=data_file_path
        )

        return download_result

    def process_files(
        self,
        download_result: ISEARDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> ISEARProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_data_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_data_dir}")

            temp_db = duckdb.connect(str(temp_data_dir / "temp.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_data_dir / 'metadata.db'}"
            )

            logger.info(msg="Processing - Merging files using duckdb")

            temp_db.sql(
                f"""
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        MYKEY AS id,
                        ID AS subject_id,
                        SIT AS text,
                        map_extract(
                            MAP {{
                                '1': 'joy',
                                '2': 'fear',
                                '3': 'anger',
                                '4': 'sadness',
                                '5': 'disgust',
                                '6': 'shame',
                                '7': 'guilt'
                                }},
                            EMOT
                        )[1] AS emotion,
                    FROM read_csv('{download_result.data_file_path}',
                        header = true,
                        delim = '|',
                        ignore_errors = true
                    )
                )
                """
            )

            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        id,
                        subject_id,
                        text,
                        IFNULL(anger, FALSE) AS anger,
                        IFNULL(disgust, FALSE) AS disgust,
                        IFNULL(fear, FALSE) AS fear,
                        IFNULL(guilt, FALSE) AS guilt,
                        IFNULL(joy, FALSE) AS joy,
                        IFNULL(sadness, FALSE) AS sadness,
                        IFNULL(shame, FALSE) AS shame
                    FROM (
                        PIVOT temp
                        ON emotion
                        USING arbitrary(true)
                    )
                    ORDER BY id
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

        processing_result = ISEARProcessingResult(
            temp_dir=temp_data_dir,
            data_dir=data_subdir,
        )

        return processing_result

    def teardown(
        self,
        download_result: ISEARDownloadResult,
        processing_result: ISEARProcessingResult,
        storage_options: dict,
    ) -> None:
        download_result.data_file_path.unlink()

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_dir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
