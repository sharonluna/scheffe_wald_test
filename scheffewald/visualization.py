import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import numpy as np
from multiple_comparisons import MultinomialComparison

def plot_test_statistics(
        results: MultinomialComparison, 
        figsize:Tuple =(5, 5),
        colors:dict = None,
        markers:dict = None
):
    """Plot test statistics vs effect sizes for multiple comparison methods"""
    
    if colors is None:
        colors = {
            'Scheffé-Wald': '#1f77b4',  # blue
            'Bonferroni': '#2ca02c',    # green
            'Tukey': '#ff7f0e'          # orange
        }
    
    if markers is None:
        markers = {
            'Scheffé-Wald': 'o',
            'Bonferroni': 's',
            'Tukey': '^'
        }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for method in results:
        # Get absolute differences for x-axis
        differences = [abs(comp[2]) for comp in method.comparisons]  # comp[2] is the difference
        
        # Create y-values as repeated test statistic
        y_values = [method.test_statistic] * len(differences)
        
        ax.scatter(
            differences,
            y_values,
            label=method.method,
            alpha=0.7,
            color=colors[method.method],
            marker=markers[method.method],
            s=100  # marker size
        )
    
    ax.set_xlabel('|Difference in Proportions|')
    ax.set_ylabel('Test Statistic')
    ax.set_title('Test Statistics vs Effect Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def visualize_comparisons(
    method_results: Dict[str, List[Tuple[int, int, float, float, float]]],
    labels: List[str],
    methods: List[str] = ['Scheffe', 'Bonferroni', 'Tukey'],
    alpha: float = 0.05,
    title: Optional[str] = None
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
    title : str, optional
        Custom title for the plot
    """
    num_comparisons = len(labels)
    indices = np.arange(num_comparisons)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = {'Scheffe': '#FF6B6B', 'Bonferroni': '#4ECDC4', 'Tukey': '#45B7D1'}
    offsets = {'Scheffe': -width, 'Bonferroni': 0, 'Tukey': width}

    for method in methods:
        if method not in method_results:
            continue
        comparisons = method_results[method]
        diffs = [comp[2] for comp in comparisons]
        ci_lowers = [comp[3] for comp in comparisons]
        ci_uppers = [comp[4] for comp in comparisons]
        errors = [[diff - lower, upper - diff] for diff, lower, upper in zip(diffs, ci_lowers, ci_uppers)]
        errors = np.array(errors).T

        ax.errorbar(
            indices + offsets[method],
            diffs,
            yerr=errors,
            fmt='o',
            color=colors.get(method, 'black'),
            label=method,
            capsize=5,
            markersize=8,
            elinewidth=2,
            capthick=2
        )

    ax.axhline(0, color='grey', linestyle='--', alpha=0.5)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Pairwise Comparisons', fontsize=12)
    ax.set_ylabel('Difference in Proportions', fontsize=12)
    ax.set_title(title or 'Confidence Intervals for Pairwise Comparisons', 
                fontsize=14, pad=20)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add alpha level reference lines
    if alpha:
        ax.text(
            ax.get_xlim()[1], 0.02,
            f'α = {alpha}',
            verticalalignment='bottom',
            horizontalalignment='right'
        )
    
    plt.tight_layout()
    return fig, ax
