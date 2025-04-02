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

GOODNEWSEVERYONE_METADATA = DatasetMetadata(
    description="The Sotck Emotions dataset, as processed by 'emotion_datasets'. This dataset consists of comments collected from StockTwits, a financial social media platform, during the COVID19 pandemic.",
    citation=(
        "@inprocedings{emotion_dataset_good_news_everyone,"
        "\n author = {Laura Bostan, Evgeny Kim, Roman Klinger},"
        "\n title = {Good News Everyone: A Corpus of News Headlines Annotated with \\\\ Emotions, Semantic Roles and Reader Perception},"
        "\n booktitle = {Proceedings of the Twelfth International Conference on Language Resources and Evaluation (LREC 2020)},"
        "\n year = {2020},"
        "\n month = {may},"
        "\n date = {11-16},"
        "\n language = {english},"
        "\n location = {Marseille, France},"
        "\n note = {preprint available at \\url{https://arxiv.org/abs/1912.03184}},"
        "\n}"
    ),
    homepage="https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/goodnewseveryone/",
    license="CC-BY 4.0",
    emotions=[
        "intensity",
        "anger",
        "annoyance",
        "disgust",
        "fear",
        "guilt",
        "joy",
        "love_including_like",
        "negative_anticipation_including_pessimism",
        "negative_surprise",
        "positive_anticipation_including_optimism",
        "positive_surprise",
        "pride",
        "sadness",
        "shame",
        "trust",
    ],
    multilabel=False,
    continuous=False,
    system="Extended Plutchik",
    domain="News headlines",
)


@dataclasses.dataclass
class GoodNewsEveryoneDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    tsv_file: pathlib.Path


@dataclasses.dataclass
class GoodNewsEveryoneProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path
    temp_dir: pathlib.Path


@dataclasses.dataclass
class GoodNewsEveryoneProcessor(DatasetBase):
    name: str = "GoodNewsEveryone"

    url: str = "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/goodnewseveryone/goodnewseveryone-v1.0.zip"

    metadata: typing.ClassVar[DatasetMetadata] = GOODNEWSEVERYONE_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> GoodNewsEveryoneDownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        file_name = pathlib.Path(self.url).name
        file_path = downloads_subdir / file_name

        logger.info(f"Download - Downloading data zip file: {file_name}")

        try:
            download(
                url=self.url,
                file_path=file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download hurricane file ({file_name}). Encountered the following exception: {e}"
            )

        with zipfile.ZipFile(file_path, "r") as f:
            f.extractall(downloads_subdir)

        tsv_file = (
            downloads_subdir / file_path.stem / "gne-release-v1.0.tsv"
        ).resolve()

        assert tsv_file.exists()

        logger.info(f"Download - Unzipped downloaded file: {file_name}")

        download_result = GoodNewsEveryoneDownloadResult(
            downloads_subdir=downloads_subdir,
            tsv_file=tsv_file,
        )

        return download_result

    def process_files(
        self,
        download_result: GoodNewsEveryoneDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> GoodNewsEveryoneProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_dir = (data_dir / tmpdirname).resolve()

            logger.debug(f"Processing - Storing intermediate files in: {temp_dir}")

            temp_db = duckdb.connect(str(temp_dir / "temp.db"))

            logger.debug(
                f"Processing - Created duckdb database in: {temp_dir / 'temp.db'}"
            )

            logger.info(msg="Processing - Ingesting data using duckdb")

            temp_db.sql(
                f"""
                CREATE OR REPLACE TABLE temp
                AS
                    SELECT
                        id,
                        source,
                        headline AS text,
                        dominant_emotion,
                        intensity,
                    FROM READ_CSV('{str(download_result.tsv_file)}',
                        header = true
                    )
                """
            )

            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS
                    SELECT
                        id,
                        source,
                        text,
                        intensity,
                        CAST(anger AS BOOL) AS anger,
                        CAST(annoyance AS BOOL) AS annoyance,
                        CAST(disgust AS BOOL) AS disgust,
                        CAST(fear AS BOOL) AS fear,
                        CAST(guilt AS BOOL) AS guilt,
                        CAST(joy AS BOOL) AS joy,
                        CAST(love_including_like AS BOOL) AS love_including_like,
                        CAST(negative_anticipation_including_pessimism AS BOOL) AS negative_anticipation_including_pessimism,
                        CAST(negative_surprise AS BOOL) AS negative_surprise,
                        CAST(positive_anticipation_including_optimism AS BOOL) AS positive_anticipation_including_optimism,
                        CAST(positive_surprise AS BOOL) AS positive_surprise,
                        CAST(pride AS BOOL) AS pride,
                        CAST(sadness AS BOOL) AS sadness,
                        CAST(shame AS BOOL) AS shame,
                        CAST(trust AS BOOL) AS trust
                    FROM (
                        PIVOT temp
                        ON dominant_emotion
                    )
                """
            )

            logger.info("Processing - Ingested data files")

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

        processing_result = GoodNewsEveryoneProcessingResult(
            data_subdir=data_subdir, temp_dir=temp_dir
        )

        return processing_result

    def teardown(
        self,
        download_result: GoodNewsEveryoneDownloadResult,
        processing_result: GoodNewsEveryoneProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
