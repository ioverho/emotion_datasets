import tempfile
import typing
import dataclasses
import os
import pathlib
import logging
import shutil
import zipfile

import duckdb
import datasets

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadError,
    DownloadResult,
    ProcessingResult,
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


SEMEVAL_2018_INTENSITY_METADATA = DatasetMetadata(
    description="The SemEval 2018 Task 1 Intensity dataset, as processed using 'emotion_datasets'. This dataset includes an array of subtasks where automatic systems have to infer the affectual state of a person from their tweet. This subset contains annotations for intensity of four basic emotions, and valence.",
    citation=(
        "@inproceedings{emotion_dataset_semeval_2018_intensity,"
        "\n    title = '{S}em{E}val-2018 Task 1: Affect in Tweets',"
        "\n    author = 'Mohammad, Saif  and"
        "\n      Bravo-Marquez, Felipe  and"
        "\n      Salameh, Mohammad  and"
        "\n      Kiritchenko, Svetlana',"
        "\n    editor = 'Apidianaki, Marianna  and"
        "\n      Mohammad, Saif M.  and"
        "\n      May, Jonathan  and"
        "\n      Shutova, Ekaterina  and"
        "\n      Bethard, Steven  and"
        "\n      Carpuat, Marine',"
        "\n    booktitle = 'Proceedings of the 12th International Workshop on Semantic Evaluation',"
        "\n    month = jun,"
        "\n    year = '2018',"
        "\n    address = 'New Orleans, Louisiana',"
        "\n    publisher = 'Association for Computational Linguistics',"
        "\n    url = 'https://aclanthology.org/S18-1001/',"
        "\n    doi = '10.18653/v1/S18-1001',"
        "\n    pages = '1--17',"
        "\n}"
    ),
    homepage="https://competitions.codalab.org/competitions/17751#learn_the_details-overview",
    license="",
    emotions=[
        "anger",
        "fear",
        "joy",
        "sadness",
        "valence",
    ],
    multilabel=True,
    continuous=True,
    system="Intensity scores for basic emotions",
    domain="Twitter posts",
)


@dataclasses.dataclass
class Semeval2018IntensityDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    downloads_extracted_subdir: pathlib.Path


@dataclasses.dataclass
class Semeval2018IntensityProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_subdir: pathlib.Path


@dataclasses.dataclass(kw_only=True)
class Semeval2018IntensityProcessor(DatasetBase):
    name: str = "Semeval2018Intensity"

    zip_url: str = "http://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"

    metadata: typing.ClassVar[DatasetMetadata] = SEMEVAL_2018_INTENSITY_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> Semeval2018IntensityDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(downloads_subdir, exist_ok=True)

        file_path = downloads_subdir / "data.zip"

        logger.info(f"Download - Downloading zip file: {file_path}")
        logger.info(f"Download - URL: {self.zip_url}")

        try:
            download(
                url=self.zip_url,
                file_path=file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download hurricane file ({file_path}). Encountered the following exception: {e}"
            )

        # Unzip the provided zip archive
        with zipfile.ZipFile(file_path, "r") as f:
            f.extractall(downloads_subdir)

        logger.info(msg=f"Download - Unzipped files to {downloads_subdir}")

        # Check for the file structure
        downloads_extracted_subdir = (
            downloads_subdir / "SemEval2018-Task1-all-data" / "English"
        )

        if not downloads_extracted_subdir.is_dir():
            raise DownloadError(
                f"Extracted zip file does not have the expected file structure. Expecting a non-empty directory at: {str(downloads_extracted_subdir.resolve())}"
            )

        download_result = Semeval2018IntensityDownloadResult(
            downloads_subdir=downloads_subdir,
            downloads_extracted_subdir=downloads_extracted_subdir,
        )

        return download_result

    def process_files(
        self,
        download_result: Semeval2018IntensityDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> Semeval2018IntensityProcessingResult:
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

            intensity_pattern = (
                download_result.downloads_extracted_subdir / "EI-reg/*/*.txt"
            )

            # Ingest the intensity files
            temp_db.sql(
                f"""
                CREATE OR REPLACE TABLE temp_intensity
                AS (
                    SELECT *
                    FROM read_csv('{str(intensity_pattern)}',
                        header = true
                    )
                )
                """
            )

            # Pivot the table
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp_intensity
                AS (
                    SELECT
                        ID AS id,
                        Tweet AS text,
                        anger,
                        fear,
                        joy,
                        sadness
                    FROM (
                        PIVOT temp_intensity
                        ON "Affect Dimension"
                        USING MEAN("Intensity Score")
                    )
                )
                """
            )

            # Ingest the valence file
            valence_pattern = download_result.downloads_extracted_subdir / "V-reg/*.txt"

            temp_db.sql(
                f"""
                CREATE OR REPLACE TABLE temp_valence
                AS (
                    SELECT *
                    FROM read_csv('{str(valence_pattern)}',
                        header = true
                    )
                )
                """
            )

            # Pivot the table
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp_valence
                AS (
                    SELECT
                        ID AS id,
                        Tweet AS text,
                        valence
                    FROM (
                        PIVOT temp_valence
                        ON "Affect Dimension"
                        USING MEAN("Intensity Score")
                    )
                )
                """
            )

            # Merge the tables together
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        temp_intensity.id,
                        temp_intensity.text,
                        valence,
                        anger,
                        fear,
                        joy,
                        sadness
                    FROM temp_intensity
                    FULL JOIN temp_valence
                    ON temp_intensity.id = temp_valence.id AND temp_intensity.text = temp_valence.text
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
            dataset_metadata=self.metadata,
        )

        update_samples(
            data_subdir=data_subdir,
            dataset_name=self.name,
            storage_options=storage_options,
        )

        logger.info("Processing - Finished dataset processing.")

        processing_result = Semeval2018IntensityProcessingResult(
            temp_dir=temp_dir,
            data_subdir=data_subdir,
        )

        return processing_result

    def teardown(
        self,
        download_result: Semeval2018IntensityDownloadResult,
        processing_result: Semeval2018IntensityProcessingResult,
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
