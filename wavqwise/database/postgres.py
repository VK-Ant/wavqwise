from wavqwise.core.base import BaseDBConnector
class PostgreSQLConnector(BaseDBConnector):
    def __init__(self, connection_string="", **kwargs):
        try:
            from sqlalchemy import create_engine
            self._engine = create_engine(connection_string)
        except ImportError:
            raise ImportError("pip install wavqwise[database]")
    def load(self, query=None, table=None, **kwargs):
        import pandas as pd
        if query: return pd.read_sql(query, self._engine)
        if table: return pd.read_sql_table(table, self._engine)
    def save(self, data, table="results", **kwargs):
        data.to_sql(table, self._engine, if_exists="replace", index=False)
