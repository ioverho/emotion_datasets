<h1 align="center">Emotion Datasets
</h1>

An effort to automate the downloading and processing of textual datasets for emotion classification. Inspired by [`sarnthil/unify-emotion-datasets`](https://github.com/sarnthil/unify-emotion-datasets/tree/master), but updated and more comprehensive. All datasets produce a [HuggingFace `datasets`](https://huggingface.co/docs/datasets/en/index) arrow dataset, and optionally some metadata files.

Currently implemented datasets:

| Name                                                                                         | System                                                           |   Labels | Multilabel   | Continuous   | Size   | Domain                                            |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------|----------|--------------|--------------|--------|---------------------------------------------------|
| [AffectiveText](https://web.eecs.umich.edu/~mihalcea/downloads.html#affective)               | Continuous ratings for different emotion classes                 |        7 | ✓            | ✓            | 1.3k   | News headlines                                    |
| [CARER](https://github.com/dair-ai/emotion_dataset)                                          | Hashtags in Twitter posts corresponding to Ekman's core emotions |        0 |              |              | 20k    | Twitter posts                                     |
| CrowdFlower                                                                                  | Hashtags in twitter posts                                        |       13 |              |              | 40k    | Twitter posts                                     |
| [EmoBank](https://github.com/JULIELab/EmoBank/tree/master)                                   | Valence-Arousal-Dominance                                        |        3 |              | ✓            | 10k    | Varied                                            |
| [EmoInt](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)                  | Subset of common emotions anotated using best-worst scaling      |        4 | ✓            | ✓            | 6.9k   | Tweets                                            |
| [FBValenceArousal](https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal) | Valence Arousal                                                  |        2 |              | ✓            | 2.9k   | Facebook posts                                    |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)      | Custom hierarchical emotion system                               |       28 |              |              | 58k    | Reddit posts                                      |
| [SentimentalLIAR](https://github.com/UNHSAILLab/SentimentalLIAR)                             | Automated emotion annotation using Google and IBM NLP APIs       |        6 | ✓            | ✓            | 13k    | Short snippets from politicians and famous people |
| [SSEC](https://www.romanklinger.de/ssec/)                                                    | A mixture between Plutchik and Ekman                             |        8 | ✓            |              | 4.8k   | Twitter posts                                     |
| [TalesEmotions](http://people.rc.rit.edu/~coagla/affectdata/index.html)                      | Ekman basic emotions                                             |        7 |              |              | 15k    | Fairy tales                                       |
| [XED](https://github.com/Helsinki-NLP/XED/tree/master)                                       | Plutchik core emotions                                           |        9 | ✓            | ✓            | 27k    | Subtitles                                         |

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

### Processing All Datasets Using Default Parameters

To simply use the default parameters, simply run the `get_all_datasets.sh` script.

### Manually Processing a Dataset

To process a single datasetr, using `uv`, run:
```sh
uv run process_dataset dataset=${DATASET}
```

The very first run might take longer, because the necessary dependencies will need to be installed.

The script has been equiped with a `hydra` CLI. Use `--help` to see which options are available.

To change the location of the output directory, run the script with the `file_system.output_dir=${OUTPUT_DIR}` command.

If the dataset has already been processed and currently resides in the output directory, the script will fail, unless `overwrite=True` is set.

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

| Name                                                                                         | Description                       |
| -------------------------------------------------------------------------------------------- | --------------------------------- |
| [SemEval-2018 Task 1: Affect in Tweets](https://competitions.codalab.org/competitions/17751) | Continuation of EmoInt            |
| [Electoral Tweets](http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html)        | Yet another Saif Mohammad dataset |

</details>

<details>
<summary>Excluded Datasets</summary>

| Name                                                                                     | Description | Exclusion Reason                                |
| ---------------------------------------------------------------------------------------- | ----------- | ----------------------------------------------- |
| [SemEval-2019 Task 3: EmoContext](https://competitions.codalab.org/competitions/19790)   |             | Emotion spread out over long context            |
| [Grounded Emotion](https://web.eecs.umich.edu/~mihalcea/downloads.html#GroundedEmotions) |             | SoTA classifiers cannot beat random performance |

</details>
