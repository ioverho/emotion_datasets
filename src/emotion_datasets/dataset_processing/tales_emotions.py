import csv
import shutil
import typing
import dataclasses
import os
import pathlib
import logging
import tarfile

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

TALES_EMOTION_METADATA = DatasetMetadata(
    description="The tales-emotions dataset, as processed by 'emotion_datasets'. Fairy tales annotated with sentence level emotions, by Cecilia Ovesdotter Alm.",
    citation=(
        "@inproceedings{emotion_dataset_tales_emotion,"
        "\n    title = 'Emotions from Text: Machine Learning for Text-based Emotion Prediction',"
        "\n    author = 'Alm, Cecilia Ovesdotter  and"
        "\n      Roth, Dan  and"
        "\n      Sproat, Richard',"
        "\n    editor = 'Mooney, Raymond  and"
        "\n      Brew, Chris  and"
        "\n      Chien, Lee-Feng  and"
        "\n      Kirchhoff, Katrin',"
        "\n    booktitle = 'Proceedings of Human Language Technology Conference and Conference on Empirical Methods in Natural Language Processing',"
        "\n    month = oct,"
        "\n    year = '2005',"
        "\n    address = 'Vancouver, British Columbia, Canada',"
        "\n    publisher = 'Association for Computational Linguistics',"
        "\n    url = 'https://aclanthology.org/H05-1073/',"
        "\n    pages = '579--586'"
        "\n}"
    ),
    homepage="http://people.rc.rit.edu/~coagla/affectdata/index.html",
    license="GPLv3",
    emotions=[
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "sad",
        "positively surprised",
        "negatively surprised",
    ],
    multilabel=False,
    continuous=False,
    system="Ekman basic emotions",
    domain="Fairy tales",
)


@dataclasses.dataclass
class TalesEmotionsDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path
    downloaded_dirs: typing.List[pathlib.Path]


@dataclasses.dataclass
class TalesEmotionsProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path


@dataclasses.dataclass
class TalesEmotionsProcessor(DatasetBase):
    name: str = "TalesEmotions"

    urls: typing.List[str] = dataclasses.field(
        default_factory=lambda: [
            "http://people.rc.rit.edu/~coagla/affectdata/Potter.tar.gz",
            "http://people.rc.rit.edu/~coagla/affectdata/HCAndersen.tar.gz",
            "http://people.rc.rit.edu/~coagla/affectdata/Grimms.tar.gz",
        ]
    )

    primary_emotion_map: typing.Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "A": "Angry",
            "D": "Disgusted",
            "F": "Fearful",
            "H": "Happy",
            "N": "Neutral",
            "Sa": "Sad",
            "Su+": "Pos.Surprised",
            "Su-": "Pos.Surprised",
        }
    )

    mood_map: typing.Dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "A": "Angry",
            "D": "Disgusted",
            "F": "Fearful",
            "H": "Happy",
            "N": "Neutral",
            "S": "Sad",
            "+": "Pos.Surprised",
            "-": "Pos.Surprised",
        }
    )

    metadata: typing.ClassVar[DatasetMetadata] = TALES_EMOTION_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> TalesEmotionsDownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        downloaded_dirs = []
        for i, url in enumerate(self.urls):
            file_name = pathlib.Path(url).name

            logger.info(
                f"Download - Downloading author zip file {i}/{len(self.urls)}: {file_name}"
            )
            author = pathlib.Path(url).name.split(".")[0]

            downloaded_file_name = pathlib.Path(url).name

            downloaded_file_path = downloads_subdir / downloaded_file_name

            try:
                download(
                    url=url,
                    file_path=downloaded_file_path,
                )
            except Exception as e:
                raise DownloadError(
                    f"Could not download zip file ({file_name}). Encountered the following exception: {e}"
                )

            logger.info(
                f"Download - Extracting author zip file {i}/{len(self.urls)}: {file_name}"
            )

            with tarfile.open(downloaded_file_path) as f:
                f.extractall(downloads_subdir / author)

            downloaded_dirs.append(downloads_subdir / author)

        download_result = TalesEmotionsDownloadResult(
            downloads_subdir=downloads_subdir, downloaded_dirs=downloaded_dirs
        )

        return download_result

    def process_files(
        self,
        download_result: TalesEmotionsDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> TalesEmotionsProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        # Process the emotion annotation files
        records = []
        for author_directory in download_result.downloaded_dirs:
            author = author_directory.name

            logger.info(f"Processing text for author: {author}")

            for mood_file_path in list(author_directory.glob("*/emmood/*.emmood")):
                with open(mood_file_path, "r", newline="\n") as f:
                    mood_file_reader = csv.reader(
                        f, delimiter="\t", quotechar="`", escapechar="\\"
                    )

                    for line in mood_file_reader:
                        sent_id = line[0].split(":")[0]

                        primary_emotion_labels = line[1].split(":")

                        primary_emotion_label_a = self.primary_emotion_map[
                            primary_emotion_labels[0]
                        ]
                        primary_emotion_label_b = self.primary_emotion_map[
                            primary_emotion_labels[0]
                        ]

                        mood_labels = line[2].split(":")

                        mood_label_a = self.mood_map[mood_labels[0]]
                        mood_label_b = self.mood_map[mood_labels[0]]

                        text = line[3]

                        records.append(
                            {
                                "author": author,
                                "story": mood_file_path.stem,
                                "sent_id": sent_id,
                                "primary_emotion_label_a": primary_emotion_label_a,
                                "primary_emotion_label_b": primary_emotion_label_b,
                                "mood_label_a": mood_label_a,
                                "mood_label_b": mood_label_b,
                                "text": text,
                            }
                        )

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

        processing_result = TalesEmotionsProcessingResult(data_subdir=data_subdir)

        return processing_result

    def teardown(
        self,
        download_result: TalesEmotionsDownloadResult,
        processing_result: TalesEmotionsProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
