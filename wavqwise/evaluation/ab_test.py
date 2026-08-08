"""A/B testing for model comparison."""
import numpy as np
from scipy import stats

class ABTest:
    @staticmethod
    def compare(scores_a, scores_b, test="ttest", alpha=0.05):
        a, b = np.array(scores_a), np.array(scores_b)
        if test == "ttest":
            stat, pvalue = stats.ttest_rel(a, b)
        elif test == "wilcoxon":
            stat, pvalue = stats.wilcoxon(a, b)
        else:
            raise ValueError(f"Unknown test: {test}")
        winner = "A" if np.mean(a) < np.mean(b) else "B"
        return {
            "statistic": stat, "p_value": pvalue,
            "significant": pvalue < alpha,
            "winner": winner,
            "mean_a": np.mean(a), "mean_b": np.mean(b),
        }
