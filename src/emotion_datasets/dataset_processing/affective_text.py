import json
import shutil
import typing
import dataclasses
import os
import pathlib
import logging
import tarfile
import re
from collections import defaultdict
import csv

import datasets

from emotion_datasets.dataset_processing.base import (
    DatasetBase,
    DownloadResult,
    ProcessingResult,
    DownloadError,
    DatasetProcessingError,
    DatasetMetadata,
)
from emotion_datasets.utils import download, get_file_stats, update_manifest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AFFECTIVE_TEXT_METADATA = DatasetMetadata(
    description="The SeMEval-2007 Task 14: Affective Text dataset, as processed by 'emotion_datasets'. Affective Text is a data set consisting of 1000 test headlines and 200 development headlines, each of them annotated with the six Eckman emotions and the polarity orientation.",
    citation=(
        "@inproceedings{10.5555/1621474.1621487,"
        "   author = {Strapparava, Carlo and Mihalcea, Rada},"
        "   title = {SemEval-2007 task 14: affective text},"
        "   year = {2007},"
        "   publisher = {Association for Computational Linguistics},"
        "   address = {USA},"
        "   abstract = {The 'Affective Text' task focuses on the classification of emotions and valence (positive/negative polarity) in news headlines, and is meant as an exploration of the connection between emotions and lexical semantics. In this paper, we describe the data set used in the evaluation and the results obtained by the participating systems.},"
        "   booktitle = {Proceedings of the 4th International Workshop on Semantic Evaluations},"
        "   pages = {70–74},"
        "   numpages = {5},"
        "   location = {Prague, Czech Republic},"
        "   series = {SemEval '07}"
        "   }"
    ),
    homepage="https://web.eecs.umich.edu/~mihalcea/downloads.html#affective",
    license="",
    emotions=[
        "anger",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "valence",
    ],
    multilabel=True,
    continuous=True,
    system="Continuous ratings for different emotion classes",
    domain="News headlines",
)


@dataclasses.dataclass
class AffectiveTextDownloadResult(DownloadResult):
    downloads_subdir: pathlib.Path


@dataclasses.dataclass
class AffectiveTextProcessingResult(
    ProcessingResult,
):
    data_subdir: pathlib.Path


CORPUS_MATCHER = re.compile(r"<instance id=\"(.*?)\">(.*?)</instance>")


@dataclasses.dataclass(kw_only=True, frozen=True)
class AffectiveTextProcessor(DatasetBase):
    name: str = "AffectiveText"

    url: str = "http://web.eecs.umich.edu/~mihalcea/downloads/AffectiveText.Semeval.2007.tar.gz"

    splits: typing.List[str] = dataclasses.field(
        default_factory=lambda: ["trial", "test"]
    )

    def get_metadata(self) -> DatasetMetadata:
        return AFFECTIVE_TEXT_METADATA

    def download_files(
        self, downloads_dir: pathlib.Path
    ) -> AffectiveTextDownloadResult:
        downloads_subdir = downloads_dir / self.name

        os.makedirs(name=downloads_subdir, exist_ok=True)

        file_name = pathlib.Path(self.url).name

        logger.info(f"Download - Downloading zip file: {file_name}")

        downloaded_file_path = downloads_dir / "affective_text.tar.gz"

        try:
            download(
                url=self.url,
                file_path=downloaded_file_path,
            )
        except Exception as e:
            raise DownloadError(
                f"Could not download zip file ({file_name}). Encountered the following exception: {e}"
            )

        logger.info("Download - Extracting zip file")

        with tarfile.open(downloaded_file_path) as f:
            f.extractall(downloads_subdir)

        download_result = AffectiveTextDownloadResult(downloads_subdir=downloads_subdir)

        return download_result

    def process_files(
        self,
        download_result: AffectiveTextDownloadResult,
        data_dir: pathlib.Path,
        overwrite: bool,
        max_shard_size: int | str,
        num_shards: typing.Optional[int],
        num_proc: typing.Optional[int],
        storage_options: dict,
    ) -> AffectiveTextProcessingResult:
        data_subdir = data_dir / self.name

        self.check_directory(data_subdir=data_subdir, overwrite=overwrite)

        records: typing.Dict[int, dict] = defaultdict(dict)  # type: ignore

        for split in self.splits:
            logger.info(f"Processing - Ingesting the {split} data")

            split_subdir = download_result.downloads_subdir / f"AffectiveText.{split}"

            if not split_subdir.exists():
                raise DatasetProcessingError(
                    f"Could not find an Affective Text split for {split}. Expecting a directory in the downloads subdir name: {split_subdir}"
                )

            # Read in the text
            with open(
                file=split_subdir / f"affectivetext_{split}.xml",
                mode="r",
            ) as f:
                text_lines = f.readlines()

                for line in text_lines:
                    if line[1:9] != "instance":
                        continue

                    # Regex process the XML file
                    # Each data instance is between <instance id=***></instance> tags
                    match: re.Match = CORPUS_MATCHER.match(line)  # type: ignore

                    line_id = int(match.group(1))
                    text = match.group(2)

                    records[line_id]["id"] = line_id
                    records[line_id]["text"] = text

            # Read in the labelled emotions
            with open(
                file=split_subdir / f"affectivetext_{split}.emotions.gold",
                mode="r",
            ) as f:
                emotion_reader = csv.reader(f, delimiter=" ")

                for line in emotion_reader:
                    line = list(map(int, line))

                    line_id = line[0]

                    # All emotion labels are scores from 0-100
                    # Renormalize to between 0-1
                    emotion_labels = {
                        "anger": int(line[1]) / 100,
                        "disgust": int(line[2]) / 100,
                        "fear": int(line[3]) / 100,
                        "joy": int(line[4]) / 100,
                        "sadness": int(line[5]) / 100,
                        "surprise": int(line[6]) / 100,
                    }

                    records[line_id].update(emotion_labels)

            # Read in the labelled emotional valence
            with open(
                file=split_subdir / f"affectivetext_{split}.valence.gold",
                mode="r",
            ) as f:
                valence_reader = csv.reader(f, delimiter=" ")

                for line in valence_reader:
                    line = list(map(int, line))

                    line_id = line[0]

                    valence_labels = {
                        "valence": int(line[1]) / 100,
                    }

                    records[line_id].update(valence_labels)

        # Convert the records to a list of dicts
        records: typing.List[dict] = list(records.values())  # type: ignore

        # Construct the HF dataset
        hf_dataset = datasets.Dataset.from_list(
            mapping=records,
            info=datasets.DatasetInfo(
                dataset_name=self.name,
                description=self.get_metadata().description,
                citation=self.get_metadata().citation,
                homepage=self.get_metadata().homepage,
                license=self.get_metadata().license,
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

        logger.info("Processing - Finished dataset processing.")

        processing_result = AffectiveTextProcessingResult(data_subdir=data_subdir)

        return processing_result

    def teardown(
        self,
        download_result: AffectiveTextDownloadResult,
        processing_result: AffectiveTextProcessingResult,
        storage_options: dict,
    ) -> None:
        shutil.rmtree(path=download_result.downloads_subdir)

        hf_dataset = datasets.Dataset.load_from_disk(
            processing_result.data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )
        hf_dataset.cleanup_cache_files()
