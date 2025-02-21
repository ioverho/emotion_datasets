import csv
import json
import typing
import dataclasses
import os
import pathlib
import logging
import shutil
import zipfile

import datasets
from tqdm import tqdm

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadResult,
    ProcessingResult,
    DatasetMetadata,
)
from emotion_datasets.utils import get_file_stats, update_manifest, update_bib_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


REN20K_METADATA = DatasetMetadata(
    description="The REN-20k dataset, as processed using 'emotion_datasets'. The Readers' Emotion News (REN) dataset is procured rom the popular online news network Rappler, where news articles were manually collected, from the year span 2014 to 2019, by checking articles with high emotion votings in the Mood Meter widget of Rappler indicating high popularity and social reach of these article.",
    citation=(
        "@article{emotion_dataset_ren20k,"
        "\n    title={REDAffectiveLM: leveraging affect enriched embedding and transformer-based neural language model for readers’ emotion detection},"
        "\n    author={Kadan, Anoop and Deepak, P and Gangan, Manjary P and Abraham, Sam Savitha and Lajish, VL},"
        "\n    journal={Knowledge and Information Systems},"
        "\n    volume={66},"
        "\n    number={12},"
        "\n    pages={7495--7525},"
        "\n    year={2024},"
        "\n    publisher={Springer}"
        "\n}"
    ),
    homepage="https://dcs.uoc.ac.in/cida/resources/ren-20k.html",
    license="CC-BY-NC",
    emotions=[
        "happy",
        "sad",
        "angry",
        "don't care",
        "inspired",
        "afraid",
        "amused",
        "annoyed",
    ],
    multilabel=True,
    continuous=True,
    system="Evoked emoions annotated by many readers",
    domain="News articles",
)


@dataclasses.dataclass
class REN20kDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    fixed_files_dir: pathlib.Path


@dataclasses.dataclass
class REN20kProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path


@dataclasses.dataclass(kw_only=True)
class REN20kProcessor(DatasetBase):
    name: str = "REN20k"

    download_file_path: pathlib.Path

    emotions: typing.ClassVar[typing.Set[str]] = {
        "happy",
        "sad",
        "angry",
        "don't care",
        "inspired",
        "afraid",
        "amused",
        "annoyed",
    }

    metadata: typing.ClassVar[DatasetMetadata] = REN20K_METADATA

    def download_files(self, downloads_dir: pathlib.Path) -> REN20kDownloadResult:
        downloads_subdir = downloads_dir / self.name
        os.makedirs(downloads_subdir, exist_ok=True)

        if self.download_file_path.is_file():
            logger.info(f"Download - Found file at {self.download_file_path}")
        else:
            raise ValueError(
                f"For {self.name} it is necessary to provide an argument 'download_file_path' that points to an existing zip file downloaded from the authors' homepage. Current value is not a file: {self.download_file_path}"
            )

        # Unzip the provided zip archive
        with zipfile.ZipFile(self.download_file_path, "r") as f:
            f.extractall(downloads_subdir)

        logger.info(msg=f"Download - Unzipped files to {downloads_subdir}")

        # Decode files from utf7 and store as utf8
        logger.info(msg="Download - Converting REN10k to utf8")

        fixed_files_dir = downloads_subdir / "fixed_files"
        os.makedirs(fixed_files_dir, exist_ok=True)

        ren10k_csv_file_paths = list(
            (downloads_subdir / "REN-20k/REN-10k/").glob("*/*.csv")
        )

        for file_path in tqdm(
            iterable=ren10k_csv_file_paths,
            desc="Encoding REN10k",
            unit="file",
        ):
            fixed_file_path = fixed_files_dir / ("ren10k_" + file_path.name)
            with open(fixed_file_path, "w", encoding="utf8") as f_out:
                with open(file_path, "r", encoding="utf8") as f_in:
                    try:
                        for line in f_in:
                            try:
                                fixed_line = line.encode(
                                    "utf8", errors="ignore"
                                ).decode("utf7", errors="ignore")

                                f_out.writelines([fixed_line])

                            except UnicodeEncodeError:
                                f_out.writelines([line])
                    except Exception as e:
                        print(
                            f"Encountered an exception. Will stop parsing this file. Exception: {e}"
                        )

        logger.info(msg="Download - Converting REN10k+ to utf8")

        ren10kplus_csv_file_paths = list(
            (downloads_subdir / "REN-20k/REN-10k+/").glob("*/*.csv")
        )

        for file_path in tqdm(
            iterable=ren10kplus_csv_file_paths,
            desc="Encoding REN10k+",
            unit="file",
        ):
            fixed_file_path = fixed_files_dir / ("ren10kplus_" + file_path.name)
            with open(fixed_file_path, "w", encoding="utf8") as f_out:
                with open(file_path, "r", encoding="utf8") as f_in:
                    try:
                        for line in f_in:
                            try:
                                fixed_line = line.encode(
                                    "utf8", errors="ignore"
                                ).decode("utf7", errors="ignore")

                                f_out.writelines([fixed_line])

                            except UnicodeEncodeError:
                                f_out.writelines([line])
                    except Exception as e:
                        print(
                            f"Encountered an exception. Will stop parsing this file. Exception: {e}"
                        )

        download_result = REN20kDownloadResult(
            downloads_subdir=downloads_subdir,
            fixed_files_dir=fixed_files_dir,
        )

        return download_result

    def process_files(
        self,
        download_result: REN20kDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> REN20kProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        logger.info(msg="Processing - Iterating over files")

        records = []
        for file_path in tqdm(
            iterable=download_result.fixed_files_dir.glob("*.csv"),
            desc="Processing",
            unit="file",
        ):
            with open(file_path, mode="r") as f:
                # Open the file
                reader = csv.DictReader(f)

                # Go through the parsed dict and try to fix keys and values
                # Skip if unable to apply fixes
                for line in reader:
                    flag = False

                    fixed_line = dict()
                    for k, v in line.items():
                        try:
                            new_key = k.strip().lower()
                            if new_key == "content":
                                new_key = "text"
                            elif new_key == "":
                                continue
                        except Exception:
                            continue

                        try:
                            if new_key in self.emotions:
                                v = float(v) / 100
                            else:
                                v = v.strip()
                        except Exception:
                            flag = True

                        fixed_line[new_key] = v

                    if not flag:
                        records.append(fixed_line)

        # Hand off to HuggingFace
        hf_dataset = datasets.Dataset.from_list(
            mapping=records,
            info=datasets.DatasetInfo(
                dataset_name=self.name,
                description=self.metadata.description,
                citation=self.metadata.citation,
                homepage=self.metadata.homepage,
                license=self.metadata.license,
            ),
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

        processing_result = REN20kProcessingResult(
            data_subdir=data_subdir,
        )

        return processing_result

    def teardown(
        self,
        download_result: REN20kDownloadResult,
        processing_result: REN20kProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
