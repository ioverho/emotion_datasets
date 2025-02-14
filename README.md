<h1 align="center">Emotion Datasets
</h1>

An effort to automate the downloading and processing of textual datasets for emotion classification. Inspired by [`sarnthil/unify-emotion-datasets`](https://github.com/sarnthil/unify-emotion-datasets/tree/master), but updated and more comprehensive. All datasets produce a [HuggingFace `datasets`](https://huggingface.co/docs/datasets/en/index) arrow dataset, and optionally some metadata files.

Currently implemented datasets:

| Name                                                                                                 | Description                                                                                           | Size | Labelling System                                                       |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ---- | ---------------------------------------------------------------------- |
| [Affective Text](https://web.eecs.umich.edu/~mihalcea/downloads.html#affective)                      | News headlines with emotion labels, as used in SemEval 2007 Task 14                                   | 1.3k | Six continuous Ekman emotions and valence                              |
| [CARER](https://github.com/dair-ai/emotion_dataset)                                                  | Self-report tweets, using only the authors' 'split' dataset configuration                             | 20k  | Six emotion classes, based on Ekman's basic emotions                   |
| Crowd Flower                                                                                         | Tweets annotated for different emotion classes. The original dataset is no longer publicly available  | 40k  | Thirteen different emotion classes                                     |
| [EmoBank](https://github.com/JULIELab/EmoBank/tree/master)                                           | Texts from various sources, annotated using VAD system. Contains the 'Affective Text' corpus as well. | 10k  | Continuous valence, arousal and dominance scores                       |
| [EmoInt](http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)                          | Tweets annotated for emotion and emotion intensity. Each tweet is annotated for at least 1 emotion.   | 6.9k | Four continuous emotion classes, most of which will be `NULL`          |
| [Facebook Valence Arousal](https://github.com/wwbp/additional_data_sets/tree/master/valence_arousal) | Facebook posts annotated for valence and arousal by two experts                                       | 2.9k | Continuous valence and arousal scores                                  |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)              | Reddit comments with manual human annotations                                                         | 58k  | Hierarchical emotion system with 28 distinct fine-grained emotions     |
| [Sentimental Liar](https://github.com/UNHSAILLab/SentimentalLIAR)                                    | A modification of the LIAR dataset, with automated labels for sentiment and some dominant emotions    | 13k  | Five emotion classes as continuous scores along with a sentiment score |
| [SSEC](https://www.romanklinger.de/ssec/)                                                            | The dataset used for SemEval 2016 Task 6, but with emotion labels                                     | 4.8k | Eight emotion classes                                                  |
| [Tales Emotions](http://people.rc.rit.edu/~coagla/affectdata/index.html)                             | Fairy tales with sentence-level annotations from two annotators for emotion and mood                  | 15k  | Eight emotion classes, based on Ekman's basic emotions                 |
|                                                                                                      |                                                                                                       |      |                                                                        |

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
    │   the processed dataset along with any metadata files
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

| Name                                                                                         | Description            |
| -------------------------------------------------------------------------------------------- | ---------------------- |
| [SemEval-2018 Task 1: Affect in Tweets](https://competitions.codalab.org/competitions/17751) | Continuation of EmoInt |
|                                                                                              |                        |

</details>

<details>
<summary>Excluded Datasets</summary>

| Name                                                                                   | Description | Exclusion Reason                     |
| -------------------------------------------------------------------------------------- | ----------- | ------------------------------------ |
| [SemEval-2019 Task 3: EmoContext](https://competitions.codalab.org/competitions/19790) |             | Emotion spread out over long context |
|                                                                                        |             |                                      |

</details>
