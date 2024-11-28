# visualization.py
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import numpy as np

def visualize_comparisons(
    method_results: Dict[str, List[Tuple[int, int, float, float, float]]],
    labels: List[str],
    methods: List[str] = ['Scheffe', 'Bonferroni', 'Tukey'],
    alpha: float = 0.05
) -> None:
    """
    Visualizes the confidence intervals from multiple methods.

    Parameters:
    method_results : dict
        A dictionary where keys are method names and values are lists of comparison tuples.
    labels : list of str
        Labels for each pairwise comparison.
    methods : list of str, optional
        Methods to include ('Scheffe', 'Bonferroni', 'Tukey')
    alpha : float, optional
        Significance level for confidence intervals (default is 0.05)
    """
    num_comparisons = len(labels)
    indices = np.arange(num_comparisons)
    width = 0.2  # Width of the error bars

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {'Scheffe': 'red', 'Bonferroni': 'blue', 'Tukey': 'green'}
    offsets = {'Scheffe': -width, 'Bonferroni': 0, 'Tukey': width}

    for method in methods:
        if method not in method_results:
            continue
        comparisons = method_results[method]
        diffs = [comp[2] for comp in comparisons]
        ci_lowers = [comp[3] for comp in comparisons]
        ci_uppers = [comp[4] for comp in comparisons]
        errors = [ [diff - lower, upper - diff] for diff, lower, upper in zip(diffs, ci_lowers, ci_uppers) ]
        errors = np.array(errors).T  # Shape (2, num_comparisons)

        ax.errorbar(
            indices + offsets[method],
            diffs,
            yerr=errors,
            fmt='o',
            color=colors.get(method, 'black'),
            label=method
        )

    ax.axhline(0, color='grey', linestyle='--')
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('Pairwise Comparisons')
    ax.set_ylabel('Difference in Proportions')
    ax.set_title('Confidence Intervals for Pairwise Comparisons')
    ax.legend()
    plt.tight_layout()
    plt.show()
