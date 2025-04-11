import csv
import dataclasses
import os
import pathlib
import logging
import shutil
import tempfile
import typing
import zipfile
import html

import duckdb
import datasets
import pandas as pd

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

TEC_METADATA = DatasetMetadata(
    description="The Hashtag Emotion Corpus or Twitter Emotion Corpus dataset, as processed by 'emotion_datasets'. Twitter posts are found using self-reported hashtags.",
    citation=(
        "@inproceedings{emotion_dataset_tec,"
        '\n    title = "{\\#}Emotional Tweets",'
        '\n    author = "Mohammad, Saif",'
        '\n    editor = "Agirre, Eneko  and'
        "\n      Bos, Johan  and"
        "\n      Diab, Mona  and"
        "\n      Manandhar, Suresh  and"
        "\n      Marton, Yuval  and"
        '\n      Yuret, Deniz",'
        '\n    booktitle = "*{SEM} 2012: The First Joint Conference on Lexical and Computational Semantics {--} Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation ({S}em{E}val 2012)",'
        '\n    month = "7-8 " # jun,'
        '\n    year = "2012",'
        '\n    address = "Montr{\'e}al, Canada",'
        '\n    publisher = "Association for Computational Linguistics",'
        '\n    url = "https://aclanthology.org/S12-1033/",'
        '\n    pages = "246--255"'
        "\n}"
    ),
    homepage="https://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html",
    license="",
    emotions=[
        "anger",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
    ],
    multilabel=False,
    continuous=False,
    system="Ekman basic emotions",
    domain="Twitter posts",
)


@dataclasses.dataclass
class TECDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    zip_path: pathlib.Path
    extracted_file: pathlib.Path


@dataclasses.dataclass
class TECProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class TECProcessor(DatasetBase):
    name: str = "TEC"

    url: str = "http://saifmohammad.com/WebDocs/Jan9-2012-tweets-clean.txt.zip"

    metadata: typing.ClassVar[DatasetMetadata] = TEC_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> TECDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(downloads_dir / self.name, exist_ok=True)

        # Download the metadata files
        zip_file_name = pathlib.Path(self.url).name
        zip_path = downloads_subdir / zip_file_name

        logger.info(f"Download - Downloading zip file: {zip_file_name}")
        logger.info(f"Download - Source: {self.url}")

        try:
            download(
                url=self.url,
                file_path=downloads_subdir / zip_file_name,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download zip file ({self.url}). Raises the following exception: {e}"
            )

        # Unzip the package
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(downloads_subdir)

        unzipped_file_name = downloads_subdir / pathlib.Path(self.url).stem

        download_result = TECDownloadResult(
            downloads_subdir=downloads_subdir,
            zip_path=zip_path,
            extracted_file=unzipped_file_name,
        )

        return download_result

    def process_files(
        self,
        download_result: TECDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> TECProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        # Ingest the data
        records = []
        with open(download_result.extracted_file, mode="r") as f:
            reader = csv.reader(f, delimiter="\t")

            for line in reader:
                id = line[0]
                emotion = line[-1]

                text = "\t".join(line[1:-1])

                id = int(id[:-1])
                text = html.unescape(text.strip())
                emotion = emotion[2:].strip()

                records.append(
                    {
                        "id": id,
                        "text": text,
                        "emotion": emotion,
                    }
                )

        # Move records to a Pandas DataFrame
        df = pd.DataFrame.from_records(records)  # noqa: F841

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_data_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_data_dir}")

            temp_db = duckdb.connect(str(temp_data_dir / "temp.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_data_dir / 'metadata.db'}"
            )

            logger.info(msg="Processing - Ingesting data frame using duckdb")

            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS
                    SELECT
                        id,
                        text,
                        CAST(anger AS BOOL) AS anger,
                        CAST(disgust AS BOOL) AS disgust,
                        CAST(fear AS BOOL) AS fear,
                        CAST(joy AS BOOL) AS joy,
                        CAST(sadness AS BOOL) AS sadness,
                        CAST(surprise AS BOOL) AS surprise,
                    FROM (
                        PIVOT df
                        ON emotion
                    )
                """
            )

            logger.info("Processing - Moved data to duckdb")

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

        processing_result = TECProcessingResult(
            temp_dir=temp_data_dir,
            data_dir=data_subdir,
        )

        return processing_result

    def teardown(
        self,
        download_result: TECDownloadResult,
        processing_result: TECProcessingResult,
        storage_options: dict,
    ) -> None:
        download_result.zip_path.unlink()
        shutil.rmtree(download_result.downloads_subdir)

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_dir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
