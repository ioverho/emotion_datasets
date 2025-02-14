import pathlib

import requests
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1",
    "Referer": "https://www.google.com",
}


def download(url: str, file_path: str | pathlib.Path):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True, headers=HEADERS)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with logging_redirect_tqdm():
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(file_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
