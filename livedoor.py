import os
import tarfile
from dataclasses import dataclass
from urllib.request import urlretrieve
from glob import glob


@dataclass
class LivedoorNewsCorpus:
    """
    download and extract livedoor corpus.
    `get_text_and_labels` returns texts and its labels.
    """

    extract_dir: str
    download_path: str = '.'
    URL: str = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"

    @property
    def text_dir(self):
        return os.path.join(self.extract_dir, 'text')

    def download(self):
        FILENAME = os.path.basename(self.URL)
        SAVED_FILE_PATH = os.path.join(self.download_path, FILENAME)
        if os.path.exists(SAVED_FILE_PATH):
            print(f"{SAVED_FILE_PATH} already exists. download stopped")
        else:
            urlretrieve(self.URL, SAVED_FILE_PATH)

        return SAVED_FILE_PATH

    def extract(self, tar_path):
        if os.path.exists(self.extract_dir):
            print(f"{self.extract_dir} already exists. extract stopped")
            return
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(self.extract_dir)

    def download_and_extract(self):
        _tar = self.download()
        self.extract(_tar)

    @property
    def categories(self):
        if not os.path.exists(self.extract_dir):
            self.download_and_extract()

        categories = [
            name for name
            in os.listdir(self.text_dir)
            if os.path.isdir(os.path.join(self.text_dir, name))
        ]
        return sorted(categories)

    def get_text_and_labels(self):
        all_text = []
        all_label = []

        # ここらへんはタスクに応じて変えられるとよいが, 今回はこれでいいや
        for cat_name in self.categories:
            files = glob(os.path.join(
                self.text_dir, cat_name, f"{cat_name}*.txt"))
            files = sorted(files)
            bodies = [self.get_body_from_file(f) for f in files]
            labels = [cat_name] * len(bodies)
            all_text.extend(bodies)
            all_label.extend(labels)

        return all_text, all_label

    @classmethod
    def get_body_from_file(_, filename):
        """get body text from a blog file."""
        with open(filename) as text_file:
            # 0: URL, 1: timestamp
            text = text_file.readlines()[2:]
            text = [sentence.strip() for sentence in text]
            text = list(filter(lambda line: line != '', text))
            return ''.join(text)
