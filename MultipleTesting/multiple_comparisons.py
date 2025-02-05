from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import numpy as np
from scipy.stats import chi2, norm, studentized_range
import pandas as pd

@dataclass
class MultinomialComparison:
    """Standardized result format for multinomial proportion comparisons"""
    method: str                          # Name of the comparison method
    comparisons: List[Tuple[int, int, float, float, float, float]]  # (i, j, diff, ci_lower, ci_upper, p_value)
    test_statistic: Optional[float] = None  # Test statistic (if applicable)
    p_value: Optional[float] = None        # Overall p-value (if applicable)
    alpha: Optional[int] = None
    df: Optional[int] = None               # Degrees of freedom for overall test
    df_error: Optional[int] = None         # Error degrees of freedom
    df_num: Optional[int] = None           # Numerator degrees of freedom

def format_test_results(result: MultinomialComparison) -> Dict[str, List[Union[str, float]]]:
    """
    Format test results into a flat dictionary with confidence intervals for multiple comparisons.

    Parameters:
    -----------
    result : MultinomialComparison
        Test results from any of the comparison methods

    Returns:
    --------
    Dict[str, List[Union[str, float]]]
        A dictionary with the following keys:
        - 'test': Method name
        - 'p_i-p_j': Category pair identifiers
        - 'diff': Differences between proportions
        - 'conf_low': Lower confidence bounds
        - 'conf_hi': Upper confidence bounds
        - 'p_value': Individual p-values for each comparison
    """
    formatted_data = {
        'test': [],
        'p_i-p_j': [],
        'diff': [],
        'conf_low': [],
        'conf_hi': [],
        'p_value': []
    }

    for comp in result.comparisons:
        i, j = comp[0], comp[1]
        diff = comp[2]
        conf_low, conf_hi = comp[3], comp[4]
        p_value = comp[5] if len(comp) >= 6 else np.nan
        
        formatted_data['test'].append(result.method)
        formatted_data['p_i-p_j'].append(f'p_{i}-p_{j}')
        formatted_data['diff'].append(diff)
        formatted_data['conf_low'].append(conf_low)
        formatted_data['conf_hi'].append(conf_hi)
        formatted_data['p_value'].append(p_value)

    return formatted_data

def format_test_results_df(result: MultinomialComparison) -> pd.DataFrame:
    """
    Format test results into a pandas DataFrame for easier manipulation and display.

    Parameters:
    -----------
    result : MultinomialComparison
        Test results from any of the comparison methods

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all comparison results
    """
    data = format_test_results(result)
    df = pd.DataFrame(data)
    
    # Add method metadata as attributes
    df.attrs['method'] = result.method
    df.attrs['test_statistic'] = result.test_statistic
    df.attrs['overall_p_value'] = result.p_value
    df.attrs['alpha'] = result.alpha
    df.attrs['df'] = result.df
    df.attrs['df_error'] = result.df_error
    df.attrs['df_num'] = result.df_num
    
    return df

def print_test_results(results: Union[MultinomialComparison, List[MultinomialComparison]]):
    """
    Print formatted results for multiple comparison tests.

    Parameters:
    -----------
    results : Union[MultinomialComparison, List[MultinomialComparison]]
        Single test result or list of test results to display
    """
    if not isinstance(results, list):
        results = [results]

    for result in results:
        print(f"\n{'-'*80}")
        print(f"{result.method} Results:")
        print(f"{'-'*80}")
        
        # Print overall test statistics if available
        if result.p_value is not None:
            print(f"Overall p-value: {result.p_value:.4f}")
        
        if result.test_statistic is not None:
            print(f"Test statistic: {result.test_statistic:.4f}")
            
        # Collect and print degrees of freedom information
        df_parts = []
        if result.df_num is not None:
            df_parts.append(f"num={result.df_num}")
        if result.df_error is not None:
            df_parts.append(f"error={result.df_error}")
        if result.df is not None:
            df_parts.append(f"df={result.df}")
            
        if df_parts:
            print(f"Degrees of freedom: ({', '.join(df_parts)})")
            
        # Print confidence level
        conf_level = 100 * (1 - result.alpha) if result.alpha is not None else 95
        print(f"\nPairwise Comparisons ({conf_level:.1f}% Confidence Intervals):")
        print(f"{'Category i':^10} {'Category j':^10} {'Difference':^12} {'CI Lower':^10} {'CI Upper':^10} {'p-value':^10}")
        print("-" * 80)
        
        for comp in result.comparisons:
            i, j = comp[0], comp[1]
            diff = comp[2]
            lower, upper = comp[3], comp[4]
            p_val = comp[5] if len(comp) >= 6 else np.nan
            
            print(f"{i:^10d} {j:^10d} {diff:^12.4f} {lower:^10.4f} {upper:^10.4f} {p_val:^10.4f}")
        
        print(f"{'-'*80}\n")

class StatisticalTest:
    def __init__(self, n_i: List[int], alpha: float = 0.05):
        """
        Initializes the StatisticalTest class with observed counts and a default significance level.

        Parameters:
        -----------
        n_i : List[int]
            Observed counts (not proportions) in each category (length r)
        alpha : float, optional
            Default significance level for confidence intervals (default is 0.05)
            
        Raises:
        -------
        ValueError
            If the sum of counts is 0 or if any count is negative
        """
        self.n_i = np.array(n_i, dtype=float)
        
        if np.any(self.n_i < 0):
            raise ValueError("All counts must be non-negative")
            
        self.n = self.n_i.sum()
        if self.n == 0:
            raise ValueError("The sum of observed counts (n_i) must be greater than 0")
        
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be between 0 and 1")
            
        self.p_hat = self.n_i / self.n
        self.r = len(self.p_hat)
        self.alpha = alpha

    def _create_pairwise_matrix(self) -> np.ndarray:
            """
            Creates constraint matrix for all pairwise comparisons.
            
            Returns:
            --------
            np.ndarray
                Matrix with dimensions (r(r-1)/2 x r) where r is number of categories
            """
            r = self.r
            num_comparisons = (r * (r - 1)) // 2
            A = np.zeros((num_comparisons, r))
            idx = 0
            for i in range(r):
                for j in range(i + 1, r):
                    A[idx, i] = 1
                    A[idx, j] = -1
                    idx += 1
            return A

    def _get_comparison_pairs(self, A: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extracts comparison pairs from constraint matrix rows.
        
        Parameters:
        -----------
        A : np.ndarray
            Constraint matrix
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of tuples containing indices of compared categories
        """
        pairs = []
        for row in A:
            nonzero = np.nonzero(row)[0]
            if len(nonzero) == 2:
                i, j = nonzero
                if row[i] > 0:
                    pairs.append((i, j))
                else:
                    pairs.append((j, i))
        return pairs

    def scheffe_wald_test(
        self,
        A: np.ndarray,
        alpha: Optional[float] = None,
        bonferroni_correction: bool = True,
    ) -> MultinomialComparison:
        """
        Performs the Scheffé-Wald test for a multinomial distribution and computes confidence intervals.

        Parameters:
        -----------
        A : Constraint matrix A (dimensions u x r)
        alpha : Optional[float]
            Significance level for confidence intervals
        bonferroni_correction : bool
            If True, adjust alpha using Bonferroni correction

        Returns:
        --------
        MultinomialComparison
            Results including confidence intervals and p-values
        """
        if alpha is None:
            alpha = self.alpha

        if A is None:
            #A = self._create_pairwise_matrix()
            raise ValueError('Must provide a contrast matrix.')

        u = A.shape[0]  # Number of constraints

        if A.shape[1] != self.r:
            raise ValueError(f"Constraint matrix A must have {self.r} columns.")

        # Compute covariance matrix
        Sigma_p_hat = (np.diag(self.p_hat) - np.outer(self.p_hat, self.p_hat)) / self.n

        # Compute components for Wald statistic
        Ap = A @ self.p_hat
        A_Sigma_A_T = A @ Sigma_p_hat @ A.T

        # Check matrix condition
        cond_num = np.linalg.cond(A_Sigma_A_T)
        if cond_num > 1e15:
            raise np.linalg.LinAlgError(
                f"Matrix A Σ_p_hat A' is ill-conditioned (condition number: {cond_num:.2e})"
            )

        # Calculate test statistics
        inv_A_Sigma_A_T = np.linalg.inv(A_Sigma_A_T)
        W = Ap.T @ inv_A_Sigma_A_T @ Ap

        se = np.sqrt(np.diag(A_Sigma_A_T))
        alpha_adj = alpha / u if bonferroni_correction else alpha
        
        critical_value = chi2.ppf(1 - alpha_adj, u)
        margin = np.sqrt(critical_value) * se
        ci_lower = Ap - margin
        ci_upper = Ap + margin

        # Individual p-values
        individual_stats = Ap**2 / np.diag(A_Sigma_A_T)
        p_values = 1 - chi2.cdf(individual_stats, df=1)

        comparisons = [
            (i, j, Ap[k], ci_lower[k], ci_upper[k], p_values[k])
            for k, (i, j) in enumerate(self._get_comparison_pairs(A))
        ]

        return MultinomialComparison(
            method="Scheffé-Wald",
            comparisons=comparisons,
            test_statistic=W,
            p_value=1 - chi2.cdf(W, u),
            alpha=alpha,
            df=u
        )

    def bonferroni_confidence_intervals(
        self,
        alpha: Optional[float] = None
    ) -> MultinomialComparison:
        """
        Computes Bonferroni-adjusted confidence intervals and p-values for all pairwise comparisons.

        Parameters:
        -----------
        alpha : Optional[float]
            Significance level. If None, uses class alpha

        Returns:
        --------
        MultinomialComparison
            Results including confidence intervals and p-values
        """
        if alpha is None:
            alpha = self.alpha

        m = self.r * (self.r - 1) // 2  # Number of comparisons
        alpha_adj = alpha / m
        z_alpha = norm.ppf(1 - alpha_adj / 2)

        max_z = 0
        comparisons = []
        df_error = self.n - 1
        
        for i in range(self.r):
            for j in range(i + 1, self.r):
                diff = self.p_hat[i] - self.p_hat[j]
                # Standard error for difference of proportions
                se = np.sqrt(
                    (self.p_hat[i] * (1 - self.p_hat[i]) + 
                        self.p_hat[j] * (1 - self.p_hat[j])) / self.n
                )
                
                z_stat = abs(diff / se)
                max_z = max(max_z, z_stat)
                
                # Two-sided p-value with Bonferroni adjustment
                p_value_ij = min(2 * (1 - norm.cdf(z_stat)) * m, 1.0)
                
                ci_lower = diff - z_alpha * se
                ci_upper = diff + z_alpha * se
                comparisons.append((i, j, diff, ci_lower, ci_upper, p_value_ij))

        # Overall p-value based on maximum test statistic
        p_value = min(2 * (1 - norm.cdf(max_z)) * m, 1.0)

        return MultinomialComparison(
            method="Bonferroni",
            comparisons=comparisons,
            test_statistic=max_z,
            p_value=p_value,
            alpha=alpha,
            df_num=1,
            df_error=df_error
        )

    def tukey_confidence_intervals(
        self,
        alpha: Optional[float] = None
    ) -> MultinomialComparison:
        """
        Computes Tukey-adjusted confidence intervals and p-values for all pairwise comparisons.

        Parameters:
        -----------
        alpha : Optional[float]
            Significance level. If None, uses class alpha

        Returns:
        --------
        MultinomialComparison
            Results including confidence intervals and p-values
        """
        if alpha is None:
            alpha = self.alpha

        df_error = self.n - self.r
        df_num = self.r - 1
        
        # Critical value from studentized range distribution
        q_alpha = studentized_range.ppf(1 - alpha, self.r, df_error) / np.sqrt(2)

        max_q = 0
        comparisons = []
        
        for i in range(self.r):
            for j in range(i + 1, self.r):
                diff = self.p_hat[i] - self.p_hat[j]
                # Standard error for difference of proportions
                se = np.sqrt(
                    (self.p_hat[i] * (1 - self.p_hat[i]) + 
                        self.p_hat[j] * (1 - self.p_hat[j])) / self.n
                )
                
                # Tukey's Q statistic
                q_stat = abs(diff / se) * np.sqrt(2)
                max_q = max(max_q, q_stat)
                
                # Individual comparison p-value
                p_value_ij = 1 - studentized_range.cdf(q_stat, self.r, df_error)
                
                ci_lower = diff - q_alpha * se
                ci_upper = diff + q_alpha * se
                comparisons.append((i, j, diff, ci_lower, ci_upper, p_value_ij))

        # Overall p-value based on maximum studentized range statistic
        p_value = 1 - studentized_range.cdf(max_q, self.r, df_error)

        return MultinomialComparison(
            method="Tukey",
            comparisons=comparisons,
            test_statistic=max_q,
            p_value=p_value,
            alpha=alpha,
            df_num=df_num,
            df_error=df_error
        )

