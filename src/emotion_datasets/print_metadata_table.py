import tabulate

from emotion_datasets.dataset_processing.base import DATASET_REGISTRY

DATASET_SIZES = {
    "AffectiveText": "1.3k",
    "CARER": "20k",
    "CrowdFlower": "40k",
    "ElectoralTweets": "3.8k",
    "EmoBank": "10k",
    "EmoInt": "6.9k",
    "FBValenceArousal": "2.9k",
    "GoEmotions": "58k",
    "SentimentalLIAR": "13k",
    "SSEC": "4.8k",
    "TalesEmotions": "15k",
    "XED": "27k",
}

if __name__ == "__main__":
    rows = []
    for name, dataset in DATASET_REGISTRY.items():
        metadata = dataset.metadata

        rows.append(
            {
                "Name": f"[{name}]({metadata.homepage})"
                if metadata.homepage != ""
                else f"{name}",
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
