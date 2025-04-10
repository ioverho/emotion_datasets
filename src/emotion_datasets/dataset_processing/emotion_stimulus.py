import dataclasses
import os
import pathlib
import logging
import re
import shutil
import tempfile
import typing
import zipfile

import duckdb
import datasets
import mosestokenizer
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

EMOTIONSTIMULUS_METADATA = DatasetMetadata(
    description="The Emotion-Stimulus dataset, as processed by 'emotion_datasets'. This dataset contains sentences with some emotion-bearing lexical-unit, mapped to Ekman's basic emotions by volunteers.",
    citation=(
        "@inproceedings{emotion_dataset_emotion_stimulus,"
        "\n title={Detecting emotion stimuli in emotion-bearing sentences},"
        "\n author={Ghazi, Diman and Inkpen, Diana and Szpakowicz, Stan},"
        "\n booktitle={Computational Linguistics and Intelligent Text Processing: 16th International Conference, CICLing 2015, Cairo, Egypt, April 14-20, 2015, Proceedings, Part II 16},"
        "\n pages={152--165},"
        "\n year={2015},"
        "\n organization={Springer}"
        "\n}"
    ),
    homepage="https://www.eecs.uottawa.ca/~diana/resources/emotion_stimulus_data/",
    license="",
    emotions=[
        "happiness",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "shame",
    ],
    multilabel=False,
    continuous=False,
    system="Ekman basic emotions",
    domain="Emotion bearing sentences",
)


@dataclasses.dataclass
class EmotionStimulusDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    zip_path: pathlib.Path
    extracted_files_path: pathlib.Path


@dataclasses.dataclass
class EmotionStimulusProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class EmotionStimulusProcessor(DatasetBase):
    name: str = "EmotionStimulus"

    url: str = (
        "http://www.eecs.uottawa.ca/~diana/resources/emotion_stimulus_data/Dataset.zip"
    )

    with_cause_file_name: str = "Emotion Cause.txt"
    without_cause_file_name: str = "No Cause.txt"

    metadata: typing.ClassVar[DatasetMetadata] = EMOTIONSTIMULUS_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> EmotionStimulusDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(downloads_dir / self.name, exist_ok=True)

        # Download the metadata files
        zip_file_name = pathlib.Path(self.url).name

        logger.info(f"Download - Downloading zip file: {zip_file_name}")
        logger.info(f"Download - Source: {self.url}")

        zip_path = downloads_subdir / "electoral_tweets.zip"

        try:
            download(
                url=self.url,
                file_path=zip_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download zip file ({self.url}). Raises the following exception: {e}"
            )

        # Unzip the package
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(downloads_subdir)

        extracted_files_path = downloads_subdir / "Dataset"

        assert extracted_files_path.exists()

        download_result = EmotionStimulusDownloadResult(
            downloads_subdir=downloads_subdir,
            zip_path=zip_path,
            extracted_files_path=extracted_files_path,
        )

        return download_result

    def process_files(
        self,
        download_result: EmotionStimulusDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> EmotionStimulusProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        tokenizer = mosestokenizer.MosesTokenizer(
            lang="en", unescape_xml=True, refined_punct_splits=True
        )

        text_pattern = re.compile(r"(<(.*?)>(.*)<\\([^br][A-Za-z0-9]+)>)")
        cause_pattern = re.compile(r"<cause>(.*)<\\cause>")

        records = []

        # Ingest the data with cause annotations
        with open(
            download_result.extracted_files_path / self.with_cause_file_name, "r"
        ) as f:
            for line in f:
                cleaned_line = cause_pattern.sub("\\1", line.strip())

                cleaned_line = tokenizer.detokenize(
                    tokenizer.tokenize(text=cleaned_line)
                ).strip()

                matches = text_pattern.match(cleaned_line)

                label = matches.group(2).strip()  # type: ignore
                text = matches.group(3).strip()  # type: ignore

                records.append({"emotion": label, "text": text})

        # Ingest the data without cause annotation
        with open(
            download_result.extracted_files_path / self.without_cause_file_name, "r"
        ) as f:
            for line in f:
                cleaned_line = tokenizer.detokenize(
                    tokenizer.tokenize(text=line)
                ).strip()

                matches = text_pattern.match(cleaned_line)

                label = matches.group(2).strip()  # type: ignore
                text = matches.group(3).strip()  # type: ignore

                records.append({"emotion": label, "text": text})

        # Move records to a Pandas DataFrame
        df = pd.DataFrame.from_records(records)

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
                        text,
                        CAST(anger AS BOOL) AS anger,
                        CAST(disgust AS BOOL) AS disgust,
                        CAST(fear AS BOOL) AS fear,
                        CAST(happy AS BOOL) AS happy,
                        CAST(sad AS BOOL) AS sad,
                        CAST(shame AS BOOL) AS shame,
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

        processing_result = EmotionStimulusProcessingResult(
            temp_dir=temp_data_dir,
            data_dir=data_subdir,
        )

        return processing_result

    def teardown(
        self,
        download_result: EmotionStimulusDownloadResult,
        processing_result: EmotionStimulusProcessingResult,
        storage_options: dict,
    ) -> None:
        download_result.zip_path.unlink()
        shutil.rmtree(download_result.extracted_files_path)

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_dir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
