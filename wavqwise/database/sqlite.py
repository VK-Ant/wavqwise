"""SQLite connector."""
import pandas as pd
from wavqwise.core.base import BaseDBConnector

class SQLiteConnector(BaseDBConnector):
    def __init__(self, connection_string="", **kwargs):
        import sqlite3
        db_path = connection_string.replace("sqlite:///", "")
        self._conn = sqlite3.connect(db_path)

    def load(self, query=None, table=None, **kwargs):
        if query: return pd.read_sql(query, self._conn)
        if table: return pd.read_sql(f"SELECT * FROM {table}", self._conn)
        raise ValueError("Provide query= or table=")

    def save(self, data, table="results", **kwargs):
        data.to_sql(table, self._conn, if_exists="replace", index=False)
