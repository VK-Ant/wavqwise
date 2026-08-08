"""Auto-detect file type and load accordingly."""
import os
import pandas as pd
from wavqwise.loaders.base import BaseLoader


class AutoLoader(BaseLoader):
    def load(self, source: str, **kwargs) -> pd.DataFrame:
        ext = os.path.splitext(source)[1].lower()
        loaders = {
            ".csv": self._load_csv, ".tsv": self._load_tsv,
            ".json": self._load_json, ".jsonl": self._load_jsonl,
            ".parquet": self._load_parquet, ".xlsx": self._load_excel,
            ".xls": self._load_excel, ".feather": self._load_feather,
        }
        loader_fn = loaders.get(ext)
        if loader_fn is None:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader_fn(source, **kwargs)

    def _load_csv(self, p, **kw): return pd.read_csv(p, parse_dates=True, **kw)
    def _load_tsv(self, p, **kw): return pd.read_csv(p, sep="\t", parse_dates=True, **kw)
    def _load_json(self, p, **kw): return pd.read_json(p, **kw)
    def _load_jsonl(self, p, **kw): return pd.read_json(p, lines=True, **kw)
    def _load_parquet(self, p, **kw): return pd.read_parquet(p, **kw)
    def _load_excel(self, p, **kw): return pd.read_excel(p, **kw)
    def _load_feather(self, p, **kw): return pd.read_feather(p, **kw)
