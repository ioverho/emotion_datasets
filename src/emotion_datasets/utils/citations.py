import pathlib
import typing
import logging

from emotion_datasets.dataset_processing.base import DatasetMetadata

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REPO_CITATION = (
    "@software{ioverho_emotion_datasets,"
    "\n    author = {Verhoeven, Ivo},"
    "\n    license = {CC-BY-4.0},"
    "\n    title = {{emotion\_datasets}},"
    "\n    url = {https://github.com/ioverho/emotion_datasets}"
    "\n}"
    )

def update_bib_file(
    data_subdir: pathlib.Path,
    dataset_metadata: DatasetMetadata,
):
    data_dir = data_subdir.parent

    bib_file = data_dir / "citations.bib"

    if bib_file.exists():
        with open(bib_file, "r") as f:
            cur_bib_file = f.read()

        logger.debug(msg="Citations - Citations file read")

    else:
        logger.debug("Citations - Citations file does not yet exist")
        cur_bib_file = REPO_CITATION

    # Split on an empty line
    cur_citations = cur_bib_file.split("\n\n")

    dataset_citation = dataset_metadata.citation

    # Add dataset citation only if it doesn't exist yet
    if dataset_citation in cur_citations:
        return None
    else:
        cur_citations.append(dataset_citation)

    # Always make the repo citation the first :)
    updated_citations = [cur_citations[0]] + sorted(cur_citations[1:])

    updated_bib_file = "\n\n".join(updated_citations)

    with open(bib_file, "w") as f:
        f.write(updated_bib_file)

        logger.debug(msg="Citations - Wrote updated citations file to disk")
