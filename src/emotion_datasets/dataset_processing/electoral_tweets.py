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

ELECTORALTWEETS_METADATA = DatasetMetadata(
    description="The Electoral Tweets dataset, as processed by 'emotion_datasets'. A set of 2012 US presidential election tweets annotated for a number of attributes pertaining to sentiment, emotion, purpose, and style by crowdsourcing.",
    citation=(
        "@article{emotion_dataset_electoral_tweets,"
        "\n  title={Sentiment, emotion, purpose, and style in electoral tweets},"
        "\n  author={Mohammad, Saif M and Zhu, Xiaodan and Kiritchenko, Svetlana and Martin, Joel},"
        "\n  journal={Information Processing \\& Management},"
        "\n  volume={51},"
        "\n  number={4},"
        "\n  pages={480--499},"
        "\n  year={2015},"
        "\n  publisher={Elsevier}"
        "\n}"
    ),
    homepage="http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html",
    license="",
    emotions=[
        "valence",
        "arousal",
        "acceptance",
        "admiration",
        "amazement",
        "anger or annoyance or hostility or fury",
        "anticipation or  expectancy or interest",
        "calmness or serenity",
        "disappointment",
        "disgust",
        "dislike",
        "fear or apprehension or panic or terror",
        "hate",
        "indifference",
        "joy or happiness or elation",
        "like",
        "sadness or gloominess or grief or sorrow",
        "surprise",
        "trust",
        "uncertainty or indecision or confusion",
        "vigilance",
    ],
    multilabel=False,
    continuous=False,
    system="Discrete categories with some aggregated emotions",
    domain="Twitter posts",
)


@dataclasses.dataclass
class ElectoralTweetsDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    zip_path: pathlib.Path
    extracted_files_path: pathlib.Path


@dataclasses.dataclass
class ElectoralTweetsProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class ElectoralTweetsProcessor(DatasetBase):
    name: str = "ElectoralTweets"

    zipfile_url: str = "http://saifmohammad.com/WebDocs/ElectoralTweetsData.zip"

    metadata: typing.ClassVar[DatasetMetadata] = ELECTORALTWEETS_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> ElectoralTweetsDownloadResult:
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

        extracted_files_path = (
            downloads_subdir
            / "ElectoralTweetsData/Annotated-US2012-Election-Tweets/Questionnaire2"
        )

        assert extracted_files_path.exists()

        download_result = ElectoralTweetsDownloadResult(
            downloads_subdir=downloads_subdir,
            zip_path=zip_path,
            extracted_files_path=extracted_files_path,
        )

        return download_result

    def process_files(
        self,
        download_result: ElectoralTweetsDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> ElectoralTweetsProcessingResult:
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
                """
                CREATE OR REPLACE TABLE temp (
                    id INTEGER,
                    annotator_trust FLOAT,
                    text VARCHAR,
                    emotion VARCHAR,
                    valence VARCHAR,
                    arousal VARCHAR
                )
                """
            )

            # Load from batch 1
            temp_db.sql(
                """
                INSERT INTO temp
                BY NAME (
                    SELECT
                        id,
                        trust as annotator_trust,
                        tweet as text,
                        q2whatemotionchooseoneoftheoptionsfrombelowthatbestrepresentstheemotion as emotion,
                        q4ifwhenansweringq2youhavechosenanemotionfromtheotheremotionscategoryorifyouansweredq4thenpleasetellusiftheemotioninthistweetispositivenegativeorneither as valence,
                        fontcolorolivetweetertweetfontbrq5howstronglyistheemotionbeingexpressedinthistweet as arousal,
                    FROM read_csv('/home/ioverho/emotion_datasets/klad/ElectoralTweets/downloads/ElectoralTweetsData/Annotated-US2012-Election-Tweets/Questionnaire2/Batch1/AnnotatedTweets.txt',
                        delim = '\t',
                        ignore_errors = true,
                        nullstr = 'BLANK'
                    )
                )
                """
            )

            # Load from batch 2
            temp_db.sql(
                """
                INSERT INTO temp
                BY NAME (
                    SELECT
                        id,
                        trust as annotator_trust,
                        tweet as text,
                        q2whatemotionchooseoneoftheoptionsfrombelowthatbestrepresentstheemotion as emotion,
                        q4ifwhenansweringq2youhavechosenanemotionfromtheotheremotionscategoryorifyouansweredq3thenpleasetellusiftheemotioninthistweetispositivenegativeorneither as valence,
                        fontcolorolivetweetertweetfontbrq5howstronglyistheemotionbeingexpressedinthistweet as arousal,
                    FROM read_csv('/home/ioverho/emotion_datasets/klad/ElectoralTweets/downloads/ElectoralTweetsData/Annotated-US2012-Election-Tweets/Questionnaire2/Batch2/AnnotatedTweets.txt',
                        delim = '\t',
                        ignore_errors = true,
                        nullstr = 'BLANK'
                    )
                )
                """
            )

            # Pivot the emotion column
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS (
                    PIVOT temp
                    ON emotion
                    USING arbitrary(true)
                )
                """
            )

            # Apply maps to the valence and arousal columns
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        id,
                        annotator_trust,
                        text,
                        valence_mapped AS valence,
                        arousal_mapped AS arousal,
                        acceptance,
                        admiration,
                        amazement,
                        \"anger or annoyance or hostility or fury\",
                        \"anticipation or  expectancy or interest\",
                        \"calmness or serenity\",
                        disappointment,
                        disgust,
                        dislike,
                        \"fear or apprehension or panic or terror\",
                        hate,
                        indifference,
                        \"joy or happiness or elation\",
                        \"like\",
                        \"sadness or gloominess or grief or sorrow\",
                        surprise,
                        trust,
                        \"uncertainty or indecision or confusion\",
                        vigilance,
                    FROM
                        temp
                        NATURAL JOIN (
                            SELECT
                                id,
                                map_extract(
                                    MAP {
                                        'neither positive nor negative': 'neutral',
                                        'negative emotion': 'negative',
                                        'positive emotion': 'positive'
                                        },
                                    valence
                                    )[1] AS valence_mapped,
                                map_extract(
                                    MAP {
                                        'the emotion is being expressed with a high intensity': 'high',
                                        'the emotion is being expressed with medium intensity': 'medium',
                                        'the emotion is being expressed with a low intensity': 'low'
                                        },
                                    arousal
                                    )[1] AS arousal_mapped
                            FROM temp
                        )
                )
                """
            )

            # Handle texts with multiple annotations
            # This makes this dataset both multilabel but not continuous
            # Each value is just a count of votes, not the insentiy of that emotion
            # Essentially, any non-zero value received at least 1 annotation
            temp_db.sql(
                """
                CREATE OR REPLACE TABLE temp
                AS (
                    SELECT
                        ARRAY_AGG(id) AS ids,
                        ARRAY_AGG(annotator_trust) AS annotator_trusts,
                        text,
                        ARRAY_AGG(valence) AS valence,
                        ARRAY_AGG(arousal) AS arousal,
                        COUNT(acceptance) AS acceptance,
                        COUNT(admiration) AS admiration,
                        COUNT(amazement) AS amazement,
                        COUNT(\"anger or annoyance or hostility or fury\") AS \"anger or annoyance or hostility or fury\",
                        COUNT(\"anticipation or  expectancy or interest\") AS \"anticipation or  expectancy or interest\",
                        COUNT(\"calmness or serenity\") AS \"calmness or serenity\",
                        COUNT(disappointment) AS disappointment,
                        COUNT(disgust) AS disgust,
                        COUNT(dislike) AS dislike,
                        COUNT(\"fear or apprehension or panic or terror\") AS \"fear or apprehension or panic or terror\",
                        COUNT(hate) AS hate,
                        COUNT(indifference) AS indifference,
                        COUNT(\"joy or happiness or elation\") AS \"joy or happiness or elation\",
                        COUNT(\"like\") AS \"like\",
                        COUNT(\"sadness or gloominess or grief or sorrow\") AS \"sadness or gloominess or grief or sorrow\",
                        COUNT(surprise) AS surprise,
                        COUNT(trust) AS trust,
                        COUNT(\"uncertainty or indecision or confusion\") AS \"uncertainty or indecision or confusion\",
                        COUNT(vigilance) AS vigilance,
                    FROM temp
                    GROUP BY text
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

        processing_result = ElectoralTweetsProcessingResult(
            temp_dir=temp_data_dir,
            data_dir=data_subdir,
        )

        return processing_result

    def teardown(
        self,
        download_result: ElectoralTweetsDownloadResult,
        processing_result: ElectoralTweetsProcessingResult,
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
