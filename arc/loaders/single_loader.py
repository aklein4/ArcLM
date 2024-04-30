
import huggingface_hub as hf

import numpy as np
import pandas as pd

from loaders.base_loader import BaseLoader


class SingleLoader(BaseLoader):

    def __init__(self, url, train, debug=False):
        super().__init__(url, train, debug)
        
        files = hf.list_repo_files(url, repo_type="dataset")
        self.parquets = [f for f in files if (f.endswith(".parquet") and self._subfolder in f)]
        if self.train:
            self.parquets = [f for f in self.parquets if f.count("train")]
        else:
            self.parquets = [f for f in self.parquets if f.count("validation")]

        self.curr_ind = 0
        self.curr_file_ind = 0
        self.done = False

        self.articles = None
        self.highlights = None
        self.load_file(0)


    def load_file(self, file_ind):
        file = self.parquets[file_ind]
        df = pd.read_parquet(f"hf://datasets/{self.url}/{file}")
        self.articles = np.array(df["article"])
        self.highlights = np.array(df["highlights"])
        assert len(self.articles) == len(self.highlights)

    def reset(self):
        self.curr_ind = 0
        self.curr_file_ind = 0
        self.done = False

        self.load_file(0)


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
                self.curr_file_ind += 1

                if self.curr_file_ind >= len(self.parquets):
                    self.curr_file_ind = 0
                    self.done = True

                self.load_file(self.curr_file_ind)

        return outicles, outlights
    