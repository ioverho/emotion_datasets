import dataclasses
import os
import pathlib
import logging
import tempfile
import typing
import shutil
import zipfile

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

USVSTHEM_METADATA = DatasetMetadata(
    description="The UsVsThem dataset, as processed by 'emotion_datasets'. Consisting of 6861 Reddit comments annotated for populist attitudes, this dataset was used to investigate the relationship between populist mindsets and social groups, as well as a range of emotions typically associated with these.",
    citation=(
        "@inproceedings{emotion_dataset_us_vs_them,"
        '\n   title = "Us vs. Them: A Dataset of Populist Attitudes, News Bias and Emotions",'
        "\n   author = \"Huguet-Cabot, Pere-Llu{'\\i}s  and"
        "\n     Abadi, David  and"
        "\n     Fischer, Agneta  and"
        '\n     Shutova, Ekaterina",'
        '\n   booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",'
        "\n   month = apr,"
        '\n   year = "2021",'
        '\n   address = "Online",'
        '\n   publisher = "Association for Computational Linguistics",'
        '\n   url = "http://dx.doi.org/10.18653/v1/2021.eacl-main.165",'
        '\n   pages = "1921--1945"'
        "\n}"
    ),
    homepage="https://github.com/LittlePea13/UsVsThem",
    license="CC BY-NC 4.0",
    emotions=[
        "anger",
        "contempt",
        "disgust",
        "fear",
        "gratitude",
        "guilt",
        "happiness",
        "hope",
        "pride",
        "relief",
        "sadness",
        "sympathy",
        "neutral",
    ],
    multilabel=True,
    continuous=False,
    system="Positive and negative emotions associated with populist attitudes",
    domain="Reddit posts",
)


@dataclasses.dataclass
class UsVsThemDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    zip_path: pathlib.Path


@dataclasses.dataclass
class UsVsThemProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class UsVsThemProcessor(DatasetBase):
    name: str = "UsVsThem"

    zipfile_url = "https://figshare.com/ndownloader/files/46885327?private_link=a2b99428c8be6c936c63"

    metadata: typing.ClassVar[DatasetMetadata] = USVSTHEM_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> UsVsThemDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(downloads_dir / self.name, exist_ok=True)

        # Download the metadata files
        zip_file_name = pathlib.Path(self.zipfile_url).name

        logger.info(f"Download - Downloading zip file: {zip_file_name}")
        logger.info(f"Download - Source: {self.zipfile_url}")

        zip_path = downloads_subdir / "electoral_tweets.zip"

        try:
            download(
                url=self.zipfile_url,
                file_path=zip_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download zip file ({self.zipfile_url}). Raises the following exception: {e}"
            )

        # Unzip the package
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(downloads_subdir)

        download_result = UsVsThemDownloadResult(
            downloads_subdir=downloads_subdir,
            zip_path=zip_path,
        )

        return download_result

    def process_files(
        self,
        download_result: UsVsThemDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> UsVsThemProcessingResult:
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

            # Just directly ingest the files
            temp_db.sql(
                f"""
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        bias,
                        "group",
                        usVSthem_scale,
                        body AS text,
                        anger AS anger,
                        contempt AS contempt,
                        disgust AS disgust,
                        fear AS fear,
                        gratitude AS gratitude,
                        guilt AS guilt,
                        happiness AS happiness,
                        hope AS hope,
                        pride AS pride,
                        relief AS relief,
                        sadness AS sadness,
                        sympathy AS sympathy,
                        "emotions_neutral" AS neutral
                    FROM read_csv('{str(download_result.downloads_subdir)}/*.csv',
                        delim = ','
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

        processing_result = UsVsThemProcessingResult(
            temp_dir=temp_data_dir,
            data_dir=data_subdir,
        )

        return processing_result

    def teardown(
        self,
        download_result: UsVsThemDownloadResult,
        processing_result: UsVsThemProcessingResult,
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
