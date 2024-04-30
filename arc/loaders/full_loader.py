
import huggingface_hub as hf

import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm

from loaders.base_loader import BaseLoader


class FullLoader(BaseLoader):

    def __init__(self, url, train, debug=False):
        super().__init__(url, train, debug)
        
        files = hf.list_repo_files(url, repo_type="dataset")
        self.parquets = [f for f in files if (f.endswith(".parquet") and self._subfolder in f)]
        if self.train:
            self.parquets = [f for f in self.parquets if f.count("train")]
        else:
            self.parquets = [f for f in self.parquets if f.count("validation")]

        articles = []
        highlights = []

        for file in tqdm(self.parquets, desc="Loading", disable=debug):

            df = pd.read_parquet(f"hf://datasets/{self.url}/{file}")
            articles.append(np.array(df["article"]))
            highlights.append(np.array(df["highlights"]))

            if debug:
                break

        self.articles = np.concatenate(articles)
        self.highlights = np.concatenate(highlights)
        assert len(self.articles) == len(self.highlights)

        self.curr_ind = 0
        self.done = False


    def reset(self):
        self.curr_ind = 0
        self.done = False


    def __len__(self):
        return len(self.articles)


    def __call__(self, batchsize):

        outicles = []
        outlights = []
        while len(outicles) < batchsize:

            outicles.append(self.articles[self.curr_ind])
            outlights.append(self.highlights[self.curr_ind])
            self.curr_ind += 1
            
            if self.curr_ind >= len(self):
                self.curr_ind = 0
                self.done = True

        return outicles, outlights
    