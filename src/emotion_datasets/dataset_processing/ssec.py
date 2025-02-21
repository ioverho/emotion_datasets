import typing
import dataclasses
import os
import pathlib
import logging
import tempfile
import shutil
import json
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
from emotion_datasets.utils import download, get_file_stats, update_manifest, update_bib_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SSEC_METADATA = DatasetMetadata(
    description="The Stance Sentiment Emotion Corpus (SSEC) dataset, as processed using 'emotion_datasets'. The SSEC corpus is an annotation of the SemEval 2016 Twitter stance and sentiment corpus with emotion labels.",
    citation=(
        "@inproceedings{emotion_dataset_ssec,"
        "\n    title={Annotation, modelling and analysis of fine-grained emotions on a stance and sentiment detection corpus},"
        "\n    author={Schuff, Hendrik and Barnes, Jeremy and Mohme, Julian and Pad{'o}, Sebastian and Klinger, Roman},"
        "\n    booktitle={Proceedings of the 8th workshop on computational approaches to subjectivity, sentiment and social media analysis},"
        "\n    pages={13--23},"
        "\n    year={2017},"
        "\n}"
    ),
    homepage="https://www.romanklinger.de/ssec/",
    license="",
    emotions=[
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ],
    multilabel=True,
    continuous=False,
    system="A mixture between Plutchik and Ekman",
    domain="Twitter posts",
)


@dataclasses.dataclass
class SSECDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    data_files: typing.List[pathlib.Path]


@dataclasses.dataclass
class SSECProcessingResult(
    ProcessingResult,
):
    temp_dir: pathlib.Path
    data_dir: pathlib.Path


@dataclasses.dataclass
class SSECProcessor(DatasetBase):
    name: str = "SSEC"

    download_url: str = "https://www.romanklinger.de/ssec/ssec-aggregated-withtext.zip"

    file_names: typing.List[str] = dataclasses.field(
        default_factory=lambda: ["train-combined-0.0.csv", "test-combined-0.0.csv"]
    )

    metadata: typing.ClassVar[DatasetMetadata] = SSEC_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> SSECDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(downloads_subdir, exist_ok=True)

        # Download zip file
        logger.info("Download - Downloading ssec zip file")
        logger.info(f"Download - Source: {self.download_url}")

        data_file_path = downloads_subdir / "ssec.zip"

        try:
            download(
                url=self.download_url,
                file_path=data_file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download SSEC zip file. Raises the following exception: {e}"
            )

        # Extract the zip archive
        with zipfile.ZipFile(data_file_path, "r") as zip_ref:
            zip_ref.extractall(downloads_subdir)

        # Keep the files we care about
        downloaded_data_files = []
        for file_name in self.file_names:
            file_path = downloads_subdir / "ssec-aggregated" / file_name
            assert file_path.exists()
            shutil.move(src=file_path, dst=downloads_subdir / file_path.name)

            downloaded_data_files.append(downloads_subdir / file_path.name)

        # Get rid of the extracted archive to avoid those files polluting dataset
        shutil.rmtree(downloads_subdir / "ssec-aggregated")

        download_result = SSECDownloadResult(
            downloads_subdir=downloads_subdir, data_files=downloaded_data_files
        )

        return download_result

    def process_files(
        self,
        download_result: SSECDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> SSECProcessingResult:
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
                    SELECT text,
                        if(anger IS NULL, false, true) AS anger,
                        if(anticipation IS NULL, false, true) AS anticipation,
                        if(disgust is NULL, false, true) AS disgust,
                        if(fear is NULL, false, true) AS fear,
                        if(joy is NULL, false, true) AS joy,
                        if(sadness is NULL, false, true) AS sadness,
                        if(surprise is NULL, false, true) AS surprise,
                        if(trust is NULL, false, true) AS trust
                    FROM read_csv('{downloaded_files_pattern}',
                        auto_detect = false,
                        delim = '\\t',
                        quote = '"',
                        escape = '"',
                        new_line = '\\n',
                        nullstr = '---',
                        header = false,
                        skip = 0,
                        ignore_errors = true,
                        columns = {{
                            'anger': 'VARCHAR',
                            'anticipation': 'VARCHAR',
                            'disgust': 'VARCHAR',
                            'fear': 'VARCHAR',
                            'joy': 'VARCHAR',
                            'sadness': 'VARCHAR',
                            'surprise': 'VARCHAR',
                            'trust': 'VARCHAR',
                            'text': 'VARCHAR'
                            }}
                        )
                )
                """
            )

            logger.info("Processing - Merged files")

            handoff_file = temp_data_dir / "output.parquet"

            temp_db.sql(f"COPY temp TO '{str(handoff_file)}' (FORMAT PARQUET);")

            logger.debug(f"Processing - Wrote to handoff file: {handoff_file}")

            hf_dataset: datasets.Dataset = datasets.Dataset.from_parquet(
                path_or_paths=str(handoff_file),
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

        update_bib_file(
            data_subdir=data_subdir,
            dataset_metadata=self.metadata,
        )

        logger.info("Processing - Finished dataset processing.")

        processing_result = SSECProcessingResult(
            temp_dir=temp_data_dir, data_dir=data_subdir
        )

        return processing_result

    def teardown(
        self,
        download_result: SSECDownloadResult,
        processing_result: SSECProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        if processing_result.temp_dir.exists():
            logger.debug("Teardown - Temp directory still exists")
            shutil.rmtree(path=processing_result.temp_dir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_dir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
