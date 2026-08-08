from wavqwise.core.base import BaseDBConnector
class MongoDBConnector(BaseDBConnector):
    def __init__(self, connection_string="", **kwargs): self._conn_str = connection_string
    def load(self, **kwargs): raise ImportError("pip install wavqwise[database]")
    def save(self, data, **kwargs): pass
