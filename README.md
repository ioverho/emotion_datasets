<h1 align="center">Emotion Datasets
</h1>

An effort to automate the downloading and processing of textual datasets for emotion classification. Inspired by [`sarnthil/unify-emotion-datasets`](https://github.com/sarnthil/unify-emotion-datasets/tree/master), but updated and more comprehensive. All datasets produce a [HuggingFace `datasets`](https://huggingface.co/docs/datasets/en/index) arrow dataset, and optionally some metadata files.

Currently implemented datasets:

| Name                                                                                         | System                                                           | Labels | Multilabel | Continuous | Size | Domain                                            |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | -----: | :--------: | :--------: | :--: | ------------------------------------------------- |
| [AffectiveText](https://web.eecs.umich.edu/~mihalcea/downloads.html#affective)               | Continuous ratings for different emotion classes                 | 7      | ✓          | ✓          | 1.3k | News headlines                                    |
| [CARER](https://github.com/dair-ai/emotion_dataset)                                          | Hashtags in Twitter posts corresponding to Ekman's core emotions | 0      |            |            | 20k  | Twitter posts                                     |
| CrowdFlower                                                                                  | Hashtags in twitter posts                                        | 13     |            |            | 40k  | Twitter posts                                     |
| [ElectoralTweets](http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html)         | Discrete categories with some aggregated emotions                | 21     | ✓          |            | 1.1k | Twitter posts                                     |
| [EmoBank](https://github.com/JULIELab/EmoBank/tree/master)                                   | Valence-Arousal-Dominance                                        | 3      |            | ✓          | 10k  | Varied                                            |
| [EmoInt](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)                  | Subset of common emotions anotated using best-worst scaling      | 4      | ✓          | ✓          | 6.9k | Twitter posts                                     |
| [FBValenceArousal](https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal) | Valence Arousal                                                  | 2      |            | ✓          | 2.9k | Facebook posts                                    |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)      | Custom hierarchical emotion system                               | 28     |            |            | 58k  | Reddit posts                                      |
| [REN20k](https://dcs.uoc.ac.in/cida/resources/ren-20k.html)[1]                               | Evoked emoions annotated by many readers                         | 8      | ✓          | ✓          | 20k  | News articles                                     |
| [SentimentalLIAR](https://github.com/UNHSAILLab/SentimentalLIAR)                             | Automated emotion annotation using Google and IBM NLP APIs       | 6      | ✓          | ✓          | 13k  | Short snippets from politicians and famous people |
| [SSEC](https://www.romanklinger.de/ssec/)                                                    | A mixture between Plutchik and Ekman                             | 8      | ✓          |            | 4.8k | Twitter posts                                     |
| [TalesEmotions](http://people.rc.rit.edu/~coagla/affectdata/index.html)                      | Ekman basic emotions                                             | 7      |            |            | 15k  | Fairy tales                                       |
| [XED](https://github.com/Helsinki-NLP/XED/tree/master)                                       | Plutchik core emotions                                           | 9      | ✓          | ✓          | 27k  | Subtitles                                         |

[1]: There are additional usage limitations in place, or the dataset is not publicly available. You are responsbile for requesting and downloading the dataset yourself from the authors' homepage.

## Installation

To install the package, first clone this repo:
```sh
git clone https://github.com/ioverho/emotion_datasets.git
```

If using [`uv`](https://docs.astral.sh/uv/), to install the most reent version of all the dependencies use sync (optional):
```sh
cd emotion_datasets
uv sync
```

## Usage

<!-- ### Processing All Datasets Using Default Parameters

To simply use the default parameters, simply run the `get_all_datasets.sh` script. -->

### Accessing Dataset Metadata

To access a dataset's metadata in a Python script, assuming you have installed this library, you can run:
```python
>>> dataset = get_dataset(${DATASET})
>>> dataset.metadata
```

Here you should replace `${DATASET}` with a dataset name. See the table above for implemented datasets. The name should not contain any spaces.

This should return a `DatasetMetadata` object that contains a description, citation and licensing information, a list of all emotion columns, and metadata on how the emotion annotations were conducted.

### Manually Processing a Dataset

To process a single datasetr, using `uv`, run:
```sh
uv run process_dataset dataset=${DATASET}
```

The script has been equiped with a `hydra` CLI. Use `--help` to see which options are available. To get help for a specific dataset, run as: `uv run process_dataset dataset=${DATASET} --help`.

To change the location of the output directory, run the script with the `file_system.output_dir=${OUTPUT_DIR}` command.

If the dataset has already been processed and currently resides in the output directory, the script will fail, unless `overwrite=True` is set.

If the data needs to be manually downloaded first (see the [1] annotation in the above table), you must set the `dataset.download_file_path` parameter to the downloaded file. This file will not be altered during processing.

### Output

Running the script for any dataset should output a directory with the following structure:
```
/data/
    ├── ${DATASET}
    │   processed data along with metadata files
    └── manifest.json
        a summary of which files can be found where
/downloads/
    Any remaining download files will reside here
/logs/
    └── ${DATASET}
        the logs produced during processing
```

All datasets are stored as HuggingFace datasets compatible directories. This implies at least 3 files:
1. `data-#####-of-#####.arrow`: the actual data, stored across arrow files
2. `dataset_info.json`: metadata relevant for users. Includes information about the homepage and citing the original dataset
3. `state.json`: metadata relevant for HuggingFace

## Appendix

<details>
<summary>WIP Datasets</summary>

| Name                                                                                         | Description                                   |
| -------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [SemEval-2018 Task 1: Affect in Tweets](https://competitions.codalab.org/competitions/17751) | Continuation of EmoInt                        |
| [VENT](https://zenodo.org/records/2537982)                                                   | Huge tweets dataset with many emotions        |
| isear                                                                                        |                                               |
| dailydialog                                                                                  |                                               |
| emotion-cause                                                                                |                                               |
| emotiondata-aman                                                                             |                                               |
| TEC                                                                                          |                                               |

### Notes

1. Both CARER and Crowdflower will need to be edited to match the same dataset schema
2. ~~Check for multilabel instaces in ElectoralTweets~~

</details>

<details>
<summary>Excluded Datasets</summary>

| Name                                                                                     | Exclusion Reason                                |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [SemEval-2019 Task 3: EmoContext](https://competitions.codalab.org/competitions/19790)   | Emotion spread out over long context            |
| [Grounded Emotion](https://web.eecs.umich.edu/~mihalcea/downloads.html#GroundedEmotions) | SoTA classifiers cannot beat random performance |

</details>
