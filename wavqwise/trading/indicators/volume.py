from wavqwise.core.base import BaseIndicator
class VWAPIndicator(BaseIndicator):
    def compute(self, data, **kwargs):
        data["VWAP"] = (data["Volume"] * (data["High"] + data["Low"] + data["Close"]) / 3).cumsum() / data["Volume"].cumsum()
        return data
