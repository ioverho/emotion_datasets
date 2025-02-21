import shutil
import tempfile
import typing
import dataclasses
import os
import pathlib
import logging
import re

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

FB_VALENCE_AROUSAL_METADATA = DatasetMetadata(
    description="The Facebook Valence Arousal dataset, as processed using 'emotion_datasets'. A data set of 2895 Social Media posts rated by two psychologically-trained annotators on two separate ordinal (valence or sentiment, and arousal or intensity) nine-point scales.",
    citation=(
        "@inproceedings{emotion_dataset_fb_valence_arousal,"
        "\n    title={Modelling valence and arousal in facebook posts},"
        "\n    author={Preo{\\c{t}}iuc-Pietro, Daniel and Schwartz, H Andrew and Park, Gregory and Eichstaedt, Johannes and Kern, Margaret and Ungar, Lyle and Shulman, Elisabeth},"
        "\n    booktitle={Proceedings of the 7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis},"
        "\n    pages={9--15},"
        "\n    year={2016}"
        "\n}"
    ),
    homepage="https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal",
    license="GPLv3",
    emotions=[
        "valence",
        "arousal",
    ],
    multilabel=False,
    continuous=True,
    system="Valence Arousal",
    domain="Facebook posts",
)


@dataclasses.dataclass
class FBValenceArousalDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    data_file_path: pathlib.Path


@dataclasses.dataclass
class FBValenceArousalProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path
    temp_dir: pathlib.Path


CORPUS_MATCHER = re.compile(r"<instance id=\"(.*?)\">(.*?)</instance>")


@dataclasses.dataclass
class FBValenceArousalProcessor(DatasetBase):
    name: str = "FBValenceArousal"

    url: str = "https://raw.githubusercontent.com/wwbp/additional_data_sets/refs/heads/master/valence_arousal/dataset-fb-valence-arousal-anon.csv"

    metadata: typing.ClassVar[DatasetMetadata] = FB_VALENCE_AROUSAL_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> FBValenceArousalDownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        file_name = pathlib.Path(self.url).name

        logger.info(f"Download - Downloading data file: {file_name}")

        downloaded_file_path = downloads_dir / "data.csv"

        try:
            download(
                url=self.url,
                file_path=downloaded_file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download data file ({file_name}). Encountered the following exception: {e}"
            )

        download_result = FBValenceArousalDownloadResult(
            downloads_subdir=downloads_subdir, data_file_path=downloaded_file_path
        )

        return download_result

    def process_files(
        self,
        download_result: FBValenceArousalDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> FBValenceArousalProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_data_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_data_dir}")

            temp_db = duckdb.connect(str(temp_data_dir / "metadata.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_data_dir / 'metadata.db'}"
            )

            logger.info(msg="Processing - Ingesting data file using duckdb")

            temp_db.sql(
                f"""
                CREATE TABLE temp
                AS
                    SELECT
                        'Anonymized Message' AS text,
                        Valence1 AS valence_1,
                        Valence2 AS valence_2,
                        Arousal1 AS arousal_1,
                        Arousal2 AS arousal_2
                    FROM (
                        SELECT *
                        FROM read_csv('{str(download_result.data_file_path)}',
                            header = true,
                            columns = {{
                                'Anonymized Message': 'VARCHAR',
                                'Valence1': 'UINTEGER',
                                'Valence2': 'UINTEGER',
                                'Arousal1': 'UINTEGER',
                                'Arousal2': 'UINTEGER'
                            }})
                    )
                """
            )

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

        processing_result = FBValenceArousalProcessingResult(
            temp_dir=temp_data_dir, data_subdir=data_subdir
        )

        return processing_result

    def teardown(
        self,
        download_result: FBValenceArousalDownloadResult,
        processing_result: FBValenceArousalProcessingResult,
        storage_options: dict,
    ) -> None:
        download_result.data_file_path.unlink()

        shutil.rmtree(path=download_result.downloads_subdir)

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
