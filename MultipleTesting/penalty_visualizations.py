import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import numpy as np


def plot_penalty_heatmap(
    proporciones: Dict[str, Dict[str, float]],
    overall_liking: Dict[str, Dict[str, float]],
    title: str = "Penalty Analysis Heatmap"
) -> None:
    """
    Creates a heatmap visualization of the penalty analysis results.
    
    Parameters:
    proporciones : dict
        Dictionary containing the proportions for each attribute and level
    overall_liking : dict
        Dictionary containing the overall liking scores
    title : str
        Title for the plot
    """
    # Prepare data for heatmap
    attributes = list(proporciones.keys())
    
    # Create matrices for proportions and differences
    prop_matrix = np.zeros((len(attributes), 3))
    ol_diff_matrix = np.zeros((len(attributes), 2))  # For TL and TM differences from JAR
    
    for i, attr in enumerate(attributes):
        prop_matrix[i] = [proporciones[attr]['TL'], 
                         proporciones[attr]['JAR'], 
                         proporciones[attr]['TM']]
        
        if attr in overall_liking:
            ol_diff_matrix[i] = [
                overall_liking[attr]['JAR'] - overall_liking[attr]['TL'],
                overall_liking[attr]['JAR'] - overall_liking[attr]['TM']
            ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1.5, 1]})
    
    # Plot proportions heatmap
    sns.heatmap(prop_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=['TL', 'JAR', 'TM'],
                yticklabels=attributes,
                ax=ax1)
    ax1.set_title('Response Proportions', pad=20)
    
    # Plot overall liking differences heatmap
    sns.heatmap(ol_diff_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                xticklabels=['TL diff', 'TM diff'],
                yticklabels=attributes,
                center=0,
                ax=ax2)
    ax2.set_title('Overall Liking Differences\n(vs JAR)', pad=20)
    
    plt.suptitle(title, fontsize=14, y=1.05)
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_penalty_scatter(
    proporciones: Dict[str, Dict[str, float]],
    overall_liking: Dict[str, Dict[str, float]],
    threshold: float = 0.20
) -> None:
    """
    Creates a scatter plot of penalties vs proportions.
    
    Parameters:
    proporciones : dict
        Dictionary containing the proportions for each attribute and level
    overall_liking : dict
        Dictionary containing the overall liking scores
    threshold : float
        Proportion threshold for considering penalties (default 0.20)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Prepare data
    tl_props = []
    tm_props = []
    tl_penalties = []
    tm_penalties = []
    attrs_tl = []
    attrs_tm = []
    
    for attr in proporciones.keys():
        if attr in overall_liking:
            # TL data
            if proporciones[attr]['TL'] > 0:
                tl_props.append(proporciones[attr]['TL'])
                tl_penalties.append(overall_liking[attr]['JAR'] - overall_liking[attr]['TL'])
                attrs_tl.append(attr)
            
            # TM data
            if proporciones[attr]['TM'] > 0:
                tm_props.append(proporciones[attr]['TM'])
                tm_penalties.append(overall_liking[attr]['JAR'] - overall_liking[attr]['TM'])
                attrs_tm.append(attr)
    
    # Plot TL scatter
    ax1.scatter(tl_props, tl_penalties, s=100)
    for i, attr in enumerate(attrs_tl):
        ax1.annotate(attr, (tl_props[i], tl_penalties[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.axvline(threshold, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Proportion Too Little')
    ax1.set_ylabel('Penalty (JAR - TL)')
    ax1.set_title('Too Little Penalties')
    ax1.grid(True, alpha=0.3)
    
    # Plot TM scatter
    ax2.scatter(tm_props, tm_penalties, s=100)
    for i, attr in enumerate(attrs_tm):
        ax2.annotate(attr, (tm_props[i], tm_penalties[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.axvline(threshold, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Proportion Too Much')
    ax2.set_ylabel('Penalty (JAR - TM)')
    ax2.set_title('Too Much Penalties')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def create_penalty_analysis_report(
    atributo: str,
    resultados_scheffe: Dict,
    proporciones: Dict[str, Dict[str, float]],
    overall_liking: Dict[str, Dict[str, float]],
    alpha: float = 0.05
) -> None:
    """
    Creates a comprehensive visual report for a single attribute.
    
    Parameters:
    atributo : str
        Name of the attribute to analyze
    resultados_scheffe : dict
        Results from the Scheffé-Wald test
    proporciones : dict
        Dictionary containing the proportions
    overall_liking : dict
        Dictionary containing the overall liking scores
    alpha : float
        Significance level
    """
    plt.style.use('seaborn')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Proportions bar plot
    ax1 = fig.add_subplot(gs[0, 0])
    props = proporciones[atributo]
    ax1.bar(['TL', 'JAR', 'TM'], [props['TL'], props['JAR'], props['TM']], 
            color=['#FF9999', '#66B2FF', '#99FF99'])
    ax1.set_title(f'Response Proportions for {atributo}')
    ax1.set_ylabel('Proportion')
    
    # Overall liking comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ol = overall_liking[atributo]
    ax2.bar(['TL', 'JAR', 'TM'], [ol['TL'], ol['JAR'], ol['TM']], 
            color=['#FF9999', '#66B2FF', '#99FF99'])
    ax2.set_title('Overall Liking Scores')
    ax2.set_ylabel('Score')
    
    # Confidence intervals from Scheffé test
    ax3 = fig.add_subplot(gs[1, :])
    intervals = resultados_scheffe['intervalos_confianza']
    comparisons = list(intervals.keys())
    
    y_pos = np.arange(len(comparisons))
    
    for i, (comp, ci) in enumerate(intervals.items()):
        mean = (ci['superior'] + ci['inferior']) / 2
        error = (ci['superior'] - ci['inferior']) / 2
        ax3.errorbar(mean, i, xerr=error, fmt='o', capsize=5, 
                    color='#4CAF50', markersize=8)
        
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(comparisons)
    ax3.set_title('Scheffé Test Confidence Intervals')
    ax3.set_xlabel('Difference in Proportions')
    
    plt.suptitle(f'Penalty Analysis Report: {atributo}', fontsize=16, y=1.02)
    plt.tight_layout()
    return fig

