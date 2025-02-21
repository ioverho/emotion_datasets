import pathlib
import json
import typing
import logging

import datasets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_samples(
    data_subdir: pathlib.Path,
    dataset_name: str,
    storage_options: typing.Optional[dict] = None,
) -> None:
    data_dir = data_subdir.parent

    if (data_dir / "samples.json").exists():
        with open(data_dir / "samples.json", "r") as f:
            samples = json.load(fp=f)

        logger.debug(msg="Samples - Samples file read")

    else:
        logger.debug("Samples - Samples file does not yet exist")
        samples = dict()

    if dataset_name not in samples:
        hf_dataset = datasets.Dataset.load_from_disk(
            dataset_path=data_subdir,
            keep_in_memory=False,
            storage_options=storage_options,
        )

        shuffled_dataset = hf_dataset.shuffle()

        sample = shuffled_dataset[0]

        samples[dataset_name] = sample
    else:
        sample = samples[dataset_name]

    with open(data_dir / "samples.json", "w") as f:
        json.dump(obj=samples, fp=f, indent=2, sort_keys=True)

        logger.debug(msg="Samples - Wrote updated samples file to disk")

    logger.info(
        f"Samples - Dataset sample: {json.dumps(obj=sample, sort_keys=True, indent=2)}"
    )
