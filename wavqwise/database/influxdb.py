from wavqwise.core.base import BaseDBConnector
class InfluxDBConnector(BaseDBConnector):
    def __init__(self, connection_string="", **kwargs): pass
    def load(self, **kwargs): raise ImportError("pip install wavqwise[database]")
    def save(self, data, **kwargs): pass
