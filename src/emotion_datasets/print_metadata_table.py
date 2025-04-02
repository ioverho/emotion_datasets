import tabulate

from emotion_datasets.dataset_processing.base import DATASET_REGISTRY

DATASET_SIZES = {
    "AffectiveText": "1.3k",
    "CancerEmo": "12k",
    "CARER": "20k",
    "CrowdFlower": "40k",
    "ElectoralTweets": "1.1k",
    "EmoBank": "10k",
    "EmoInt": "6.9k",
    "FBValenceArousal": "2.9k",
    "GoEmotions": "58k",
    "GoodNewsEveryone": "5k",
    "Hurricanes8": "14k",
    "Hurricanes24": "15k",
    "ISEAR": "7.6k",
    "REN20k": "20k",
    "Semeval2018Classification": "11k",
    "Semeval2018Intensity": "11k",
    "SentimentalLIAR": "13k",
    "SSEC": "4.8k",
    "StockEmotions": "10k",
    "TalesEmotions": "15k",
    "UsVsThem": "6.8k",
    "WASSA22": "2.1k",
    "XED": "27k",
}

NAME_ANNOTATIONS = {"REN20k": "[1]"}

if __name__ == "__main__":
    rows = []
    for name, dataset in DATASET_REGISTRY.items():
        metadata = dataset.metadata

        if metadata.homepage != "":
            dataset_name_and_url = f"[{name}]({metadata.homepage})"
        else:
            dataset_name_and_url = f"{name}"

        dataset_name_and_url += f"{NAME_ANNOTATIONS.get(name, '')}"

        rows.append(
            {
                "Name": dataset_name_and_url,
                # "Description": f"{metadata.description.split('.', maxsplit=1)[1].strip()}",
                "System": metadata.system,
                "Labels": len(metadata.emotions),
                "Multilabel": "✓" if metadata.multilabel else "",
                "Continuous": "✓" if metadata.continuous else "",
                "Size": f"{DATASET_SIZES.get(name, '')}",
                "Domain": metadata.domain,
            }
        )

    print(
        "\n"
        + tabulate.tabulate(
            rows,
            tablefmt="github",
            headers={k: k for k in rows[0]},
            colalign=["left", "left", "right", "center", "center", "center", "left"],
        )
        + "\n"
    )
