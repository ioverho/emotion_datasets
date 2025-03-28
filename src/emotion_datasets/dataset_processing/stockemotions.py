import shutil
import tempfile
import typing
import dataclasses
import os
import pathlib
import logging

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

STOCKEMOTIONS_METADATA = DatasetMetadata(
    description="The Sotck Emotions dataset, as processed by 'emotion_datasets'. This dataset consists of comments collected from StockTwits, a financial social media platform, during the COVID19 pandemic.",
    citation=(
        "@article{lee2023stockemotions,"
        "   \ntitle={StockEmotions: Discover Investor Emotions for Financial Sentiment Analysis and Multivariate Time Series},"
        "   \nauthor={Lee, Jean and Youn, Hoyoul Luis and Poon, Josiah and Han, Soyeon Caren},"
        "   \njournal={arXiv preprint arXiv:2301.09279},"
        "   \nyear={2023}"
        "\n}"
    ),
    homepage="https://github.com/adlnlp/StockEmotions/tree/main",
    license="",
    emotions=[
        "ambiguous",
        "amusement",
        "anger",
        "anxiety",
        "belief",
        "confusion",
        "depression",
        "disgust",
        "excitement",
        "optimism",
        "panic",
        "surprise",
        "sentiment",
    ],
    multilabel=False,
    continuous=False,
    system="Custom emotions set",
    domain="Social media comments about stocks",
)


@dataclasses.dataclass
class StockEmotionsDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    downloaded_file_paths: typing.List[pathlib.Path]


@dataclasses.dataclass
class StockEmotionsProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path
    temp_dir: pathlib.Path


@dataclasses.dataclass
class StockEmotionsProcessor(DatasetBase):
    name: str = "StockEmotions"

    urls: typing.List[str] = dataclasses.field(
        default_factory=lambda: [
            "https://raw.githubusercontent.com/adlnlp/StockEmotions/refs/heads/main/tweet/train_stockemo.csv",
            "https://raw.githubusercontent.com/adlnlp/StockEmotions/refs/heads/main/tweet/val_stockemo.csv",
            "https://raw.githubusercontent.com/adlnlp/StockEmotions/refs/heads/main/tweet/test_stockemo.csv",
        ]
    )

    metadata: typing.ClassVar[DatasetMetadata] = STOCKEMOTIONS_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> StockEmotionsDownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        downloaded_file_paths = []
        for i, url in enumerate(self.urls):
            file_name = pathlib.Path(url).name
            file_path = downloads_subdir / file_name

            logger.info(
                f"Download - Downloading split {i}/{len(self.urls)}: {file_name}"
            )

            try:
                download(
                    url=url,
                    file_path=file_path,
                )
            except Exception as e:
                raise DownloadError(
                    f"Could not download hurricane file ({file_name}). Encountered the following exception: {e}"
                )

            downloaded_file_paths.append(file_path)

        download_result = StockEmotionsDownloadResult(
            downloads_subdir=downloads_subdir,
            downloaded_file_paths=downloaded_file_paths,
        )

        return download_result

    def process_files(
        self,
        download_result: StockEmotionsDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> StockEmotionsProcessingResult:
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

            csv_file_pattern = str(download_result.downloads_subdir / "*.csv")

            temp_db.sql(
                f"""
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        id,
                        ticker,
                        original AS text,
                        CAST(ambiguous AS bool) AS ambiguous,
                        CAST(amusement AS bool) AS amusement,
                        CAST(anger AS bool) AS anger,
                        CAST(anxiety AS bool) AS anxiety,
                        CAST(belief AS bool) AS belief,
                        CAST(confusion AS bool) AS confusion,
                        CAST(depression AS bool) AS depression,
                        CAST(disgust AS bool) AS disgust,
                        CAST(excitement AS bool) AS excitement,
                        CAST(optimism AS bool) AS optimism,
                        CAST(panic AS bool) AS panic,
                        CAST(surprise AS bool) AS surprise,
                        senti_label AS sentiment
                    FROM (
                        PIVOT (
                            SELECT *
                            FROM READ_CSV('{csv_file_pattern}',
                                header=true
                                )
                        )
                        ON emo_label
                    )
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

        processing_result = StockEmotionsProcessingResult(
            data_subdir=data_subdir, temp_dir=temp_dir
        )

        return processing_result

    def teardown(
        self,
        download_result: StockEmotionsDownloadResult,
        processing_result: StockEmotionsProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
