# dataset.py
import json, pathlib, ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer


class BasketDataset(Dataset):
    """
    Each CSV already corresponds to one split: train.csv, valid.csv, test.csv.
    – `csv_path` : that split’s file.
    – `all_csv_paths` : list[str] of *all* split paths; used **once** to build
       the SBERT embedding cache the first time the class is instantiated.
    Lookup happens inline in __getitem__; no external helper.
    """

    def __init__(self,
                 csv_path: str,
                 all_csv_paths: list[str],
                 sbert_name: str = "all-MiniLM-L6-v2",
                 cache_dir: str = ".cache_sbert"):

        self.encoder   = SentenceTransformer(sbert_name)
        self.d         = self.encoder.get_sentence_embedding_dimension()
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_path = self.cache_dir / f"{sbert_name.replace('/','_')}.json"

        # -------- 1. build / load embedding dictionary ------------------
        if self.cache_path.exists():
            print(f"Loading embedding cache from {self.cache_path}")
            with self.cache_path.open() as f:
                self.emb_dict = {k: torch.tensor(v) for k, v in json.load(f).items()}
        else:
            self._build_embedding_cache(all_csv_paths)

        # -------- 2. read this split’s dataframe -----------------------
        df = pd.read_csv(csv_path)
        df["products_before"] = df["products_before"].apply(ast.literal_eval)
        pairs = [(basket, label) for basket, label in zip(df["products_before"], df["order_after"])
                if len(basket) > 0]

        self.baskets, self.labels = zip(*pairs)
        self.baskets = list(self.baskets)
        self.labels  = list(self.labels)

    # -------------------------------------------------------------------
    def _build_embedding_cache(self, csv_paths):
        print("Building SBERT cache …")
        unique_titles = set()
        for p in csv_paths:
            df = pd.read_csv(p)
            df["products_before"] = df["products_before"].apply(ast.literal_eval)
            unique_titles.update(df["products_before"].explode().unique())

        unique_titles = {t for t in unique_titles if isinstance(t, str)}
        titles = sorted(unique_titles)
        vecs = self.encoder.encode(titles, batch_size=512,
                                   convert_to_numpy=True, show_progress_bar=True)
        self.emb_dict = {t: torch.tensor(v) for t, v in zip(titles, vecs)}
        with self.cache_path.open("w") as f:
            json.dump({k: v.tolist() for k, v in self.emb_dict.items()}, f)

    # -------------------------------------------------------------------
    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        vecs = torch.stack([self.emb_dict[p] for p in self.baskets[idx]])  # (n_i, d)
        return vecs, torch.tensor(self.labels[idx], dtype=torch.float32)

    # -------------------------------------------------------------------
    @staticmethod
    def collate(batch):
        baskets, labels = zip(*batch)
        return list(baskets), torch.stack(labels)
