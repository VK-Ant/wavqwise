"""Evaluation metrics for time-series forecasting."""
import numpy as np

class Metrics:
    @staticmethod
    def mae(actual, predicted):
        return np.mean(np.abs(np.array(actual) - np.array(predicted)))

    @staticmethod
    def mse(actual, predicted):
        return np.mean((np.array(actual) - np.array(predicted)) ** 2)

    @staticmethod
    def rmse(actual, predicted):
        return np.sqrt(Metrics.mse(actual, predicted))

    @staticmethod
    def mape(actual, predicted):
        a, p = np.array(actual), np.array(predicted)
        mask = a != 0
        return np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100

    @staticmethod
    def smape(actual, predicted):
        a, p = np.array(actual), np.array(predicted)
        denom = (np.abs(a) + np.abs(p)) / 2
        mask = denom != 0
        return np.mean(np.abs(a[mask] - p[mask]) / denom[mask]) * 100

    @staticmethod
    def mase(actual, predicted, seasonal_period=1):
        a, p = np.array(actual), np.array(predicted)
        n = len(a)
        naive_errors = np.abs(np.diff(a, n=seasonal_period))
        scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1
        return np.mean(np.abs(a - p)) / max(scale, 1e-8)

    @staticmethod
    def coverage(actual, lower, upper):
        a = np.array(actual)
        return np.mean((a >= np.array(lower)) & (a <= np.array(upper)))

    @staticmethod
    def all_metrics(actual, predicted, lower=None, upper=None):
        result = {
            "MAE": Metrics.mae(actual, predicted),
            "RMSE": Metrics.rmse(actual, predicted),
            "MAPE": Metrics.mape(actual, predicted),
            "SMAPE": Metrics.smape(actual, predicted),
        }
        if lower is not None and upper is not None:
            result["Coverage"] = Metrics.coverage(actual, lower, upper)
        return result
