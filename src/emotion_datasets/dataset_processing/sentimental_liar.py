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

SENTIMENTAL_LIAR_METADATA = DatasetMetadata(
    description="The Sentimental LIAR dataset, as processed using 'emotion_datasets'. Sentimental LIAR is a modified and further extended version of the LIAR extension introduced by Kirilin et al. Sentiments are derived using the Google NLP API, whereas emotion scores were extracted using the IBM NLP API for each claim, which determine the detected level of 6 emotional states namely anger, sadness, disgust, fear and joy.",
    citation=(
        "@inproceedings{emotion_dataset_sentimental_liar,"
        "\n    title={Sentimental LIAR: Extended Corpus and Deep Learning Models for Fake Claim Classification},"
        "\n    author={Upadhayay, Bibek and Behzadan, Vahid},"
        "\n    booktitle={2020 IEEE International Conference on Intelligence and Security Informatics (ISI)},"
        "\n    pages={1--6},"
        "\n    year={2020},"
        "\n    organization={IEEE}"
        "\n}"
    ),
    homepage="https://github.com/UNHSAILLab/SentimentalLIAR",
    license="",
    emotions=[
        "sentiment",
        "anger",
        "fear",
        "joy",
        "disgust",
        "sad",
    ],
    multilabel=True,
    continuous=True,
    system="Automated emotion annotation using Google and IBM NLP APIs",
    domain="Short snippets from politicians and famous people",
)


@dataclasses.dataclass
class SentimentalLIARDownloadResult(DownloadResult):
    data_files: typing.List[pathlib.Path]


@dataclasses.dataclass
class SentimentalLIARProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class SentimentalLIARProcessor(DatasetBase):
    name: str = "SentimentalLIAR"

    data_files: typing.List[str] = dataclasses.field(
        default_factory=lambda: [
            "https://raw.githubusercontent.com/UNHSAILLab/SentimentalLIAR/refs/heads/master/train_final.csv",
            "https://raw.githubusercontent.com/UNHSAILLab/SentimentalLIAR/refs/heads/master/valid_final.csv",
            "https://raw.githubusercontent.com/UNHSAILLab/SentimentalLIAR/refs/heads/master/test_final.csv",
        ]
    )

    metadata: typing.ClassVar[DatasetMetadata] = SENTIMENTAL_LIAR_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> SentimentalLIARDownloadResult:
        os.makedirs(downloads_dir / self.name, exist_ok=True)

        # Data files
        downloaded_data_files = []
        for i, data_file in enumerate(self.data_files):
            data_file_name = pathlib.Path(data_file).name

            logger.info(
                f"Download - Downloading metadata file {i + 1}/{len(self.data_files)}: {data_file}"
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
                    f"Could not download SentimentalLIAR data file ({data_file}). Raises the following exception: {e}"
                )

            downloaded_data_files.append(data_file_path)

        download_result = SentimentalLIARDownloadResult(
            data_files=downloaded_data_files
        )

        return download_result

    def process_files(
        self,
        download_result: SentimentalLIARDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> SentimentalLIARProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        downloaded_files_pattern = str(download_result.data_files[0].with_stem("*"))

        logger.debug(f"Processing - Looking for files in {downloaded_files_pattern}")

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
                    SELECT ID, statement, subject, speaker, context, sentiment, sentiment_score, anger, fear, joy, disgust, sad
                    FROM read_csv('{downloaded_files_pattern}',
                        header = true,
                        union_by_name = true
                    )
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

        processing_result = SentimentalLIARProcessingResult(
            temp_dir=temp_data_dir, data_dir=data_subdir
        )

        return processing_result

    def teardown(
        self,
        download_result: SentimentalLIARDownloadResult,
        processing_result: SentimentalLIARProcessingResult,
        storage_options: dict,
    ) -> None:
        for data_file_path in download_result.data_files:
            data_file_path.unlink()

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(path=processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_dir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
