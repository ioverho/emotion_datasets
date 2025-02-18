import dataclasses
import os
import pathlib
import logging
import tempfile
import typing
import shutil
import json

import duckdb
import datasets

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadResult,
    ProcessingResult,
    DownloadError,
    DatasetMetadata,
)
from emotion_datasets.utils import download, get_file_stats, update_manifest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMOBANK_METADATA = DatasetMetadata(
    description="The EmoBank dataset, as processed using 'emotion_datasets'. EmoBank is a large-scale text corpus manually annotated with emotion according to the psychological Valence-Arousal-Dominance scheme. It was build at JULIE Lab, Jena University and is described in detail in papers from EACL 2017 and LAW 2017.",
    citation=(
        "@article{buechel2022emobank,"
        "   title={Emobank: Studying the impact of annotation perspective and representation format on dimensional emotion analysis},"
        "   author={Buechel, Sven and Hahn, Udo},"
        "   journal={arXiv preprint arXiv:2205.01996},"
        "   year={2022}"
        "}"
    ),
    homepage="https://github.com/JULIELab/EmoBank/tree/master",
    license="CC-BY-SA 4.0",
    emotions=[
        "Valence",
        "Arousal",
        "Dominance",
    ],
    multilabel=False,
    continuous=True,
    system="Valence-Arousal-Dominance",
    domain="Varied",
)


@dataclasses.dataclass
class EmoBankDownloadResult(DownloadResult):
    metadata_file_path: pathlib.Path
    data_file_path: pathlib.Path


@dataclasses.dataclass
class EmoBankProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class EmoBankProcessor(DatasetBase):
    name: str = "EmoBank"

    data_file: str = "https://raw.githubusercontent.com/JULIELab/EmoBank/refs/heads/master/corpus/emobank.csv"
    metadata_file: str = "https://raw.githubusercontent.com/JULIELab/EmoBank/refs/heads/master/corpus/meta.tsv"

    metadata: typing.ClassVar[DatasetMetadata] = EMOBANK_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> EmoBankDownloadResult:
        os.makedirs(downloads_dir / self.name, exist_ok=True)

        # Download the metadata files
        metadata_file_name = pathlib.Path(self.metadata_file).name

        logger.info(f"Download - Downloading metadata file: {metadata_file_name}")
        logger.info(f"Download - Source: {self.metadata_file}")

        metadata_file_path = downloads_dir / self.name / metadata_file_name

        try:
            download(
                url=self.metadata_file,
                file_path=metadata_file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download EmoBank metadata file ({self.metadata_file}). Raises the following exception: {e}"
            )

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

        download_result = EmoBankDownloadResult(
            metadata_file_path=metadata_file_path, data_file_path=data_file_path
        )

        return download_result

    def process_files(
        self,
        download_result: EmoBankDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> EmoBankProcessingResult:
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
                CREATE TABLE temp
                AS
                    SELECT emo_bank_data.id, document, category, subcategory, split, V, A, D, text
                    FROM (
                        SELECT *
                        FROM read_csv('{str(download_result.data_file_path)}',
                            header = true,
                            columns = {{
                                'id': 'VARCHAR',
                                'split': 'VARCHAR',
                                'V': FLOAT,
                                'A': FLOAT,
                                'D': FLOAT,
                                'text': VARCHAR,
                            }})
                    ) AS emo_bank_data INNER JOIN (
                        SELECT *
                        FROM read_csv('{str(download_result.metadata_file_path)}',
                            sep = '\\t',
                            header = true,
                            columns = {{
                                'id': 'VARCHAR',
                                'document': 'VARCHAR',
                                'category': VARCHAR,
                                'subcategory': VARCHAR,
                            }})
                    ) AS emo_bank_metadata
                    ON emo_bank_data.id = emo_bank_metadata.id
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

            shuffled_dataset = hf_dataset.shuffle()

            logger.info(
                f"Processing - Dataset sample: {json.dumps(obj=shuffled_dataset[0], sort_keys=True, indent=2)}"
            )

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

        processing_result = EmoBankProcessingResult(
            temp_dir=temp_data_dir, data_dir=data_dir / "EmoBank"
        )

        return processing_result

    def teardown(
        self,
        download_result: EmoBankDownloadResult,
        processing_result: EmoBankProcessingResult,
        storage_options: dict,
    ) -> None:
        download_result.data_file_path.unlink()
        download_result.metadata_file_path.unlink()

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_dir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
