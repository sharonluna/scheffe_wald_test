# multiple_comparisons_tests.py
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.stats import chi2, norm, studentized_range
from dataclasses import dataclass


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
    

class StatisticalTest:
    def __init__(self, n_i: List[int], alpha: float = 0.05):
        """
        Initializes the StatisticalTest class with observed counts and a default significance level.

        Parameters:
        n_i : list of int
            Observed counts in each category (length r)
        alpha : float, optional
            Default significance level for confidence intervals (default is 0.05)
        """
        self.n_i = np.array(n_i)
        self.n = self.n_i.sum()

        if self.n == 0:
            raise ValueError("The sum of observed counts (n_i) must be greater than 0.")
        
        self.p_hat = self.n_i/self.n
        self.r = len(self.p_hat)
        self.alpha = alpha

    def scheffe_wald_test(
        self,
        A: np.ndarray = None,
        alpha: Optional[float] = None,
        bonferroni_correction: bool = True,
    ) -> MultinomialComparison:
        """
        Performs the Scheffé-Wald test for a multinomial distribution and computes confidence intervals.

        Parameters:
        A : 2D array-like
            Constraint matrix A (dimensions u x r)
        alpha : float, optional
            Significance level for confidence intervals (default is the class alpha)
        bonferroni_correction : bool, optional
            If True, adjust alpha using Bonferroni correction.

        Returns:
        MultinomialComparison
            Standardized results including confidence intervals and p-values
        """
        
        if alpha is None:
            alpha = self.alpha

        if A is None:
            A = self._create_pairwise_matrix()

        u = A.shape[0]  # Number of constraints

        # Dimension check for A
        if A.shape[1] != self.r:
            raise ValueError(f"Constraint matrix A must have {self.r} columns, matching the number of categories.")

        # Compute the covariance matrix Σ_p_hat
        diag_p_hat = np.diag(self.p_hat)
        outer_p_hat = np.outer(self.p_hat, self.p_hat)
        Sigma_p_hat = (diag_p_hat - outer_p_hat)/self.n # Covariance matrix

        # Compute W = p_hat' A' (A Σ_p_hat A')^{-1} A p_hat
        Ap = A @ self.p_hat  # A * p_hat
        A_Sigma_A_T = A @ Sigma_p_hat @ A.T

        # Check if A_Sigma_A_T is invertible
        if np.linalg.matrix_rank(A_Sigma_A_T) < u:
            print(A)
            raise np.linalg.LinAlgError("Matrix A Σ_p_hat A' is singular and cannot be inverted.")
        

        # Compute inverse
        inv_A_Sigma_A_T = np.linalg.inv(A_Sigma_A_T)
        W = Ap.T @ inv_A_Sigma_A_T @ Ap  # Wald Test statistic

        # Compute standard errors
        se = np.sqrt(np.diag(A_Sigma_A_T))

        if bonferroni_correction:
            alpha_adj = alpha / u
        else:
            alpha_adj = alpha

        # Compute critical value for confidence intervals using the Scheffe adjustment
        critical_value = chi2.ppf(1 - alpha_adj, u)
            
        # Confidence intervals
        ci_lower = Ap - np.sqrt(critical_value) * se
        ci_upper = Ap + np.sqrt(critical_value) * se

        # Format results
        comparisons = [
            (i, j, Ap[k], ci_lower[k], ci_upper[k])
            for k, (i, j) in enumerate(self._get_comparison_pairs(A))
        ]

        return MultinomialComparison(
            method="Scheffé-Wald",
            comparisons=comparisons,
            test_statistic=W,
            alpha=alpha,
            p_value=1 - chi2.cdf(W, u),
            df=u
        )
    
    def bonferroni_confidence_intervals(
            self,
            alpha: Optional[float] = None
        ) -> MultinomialComparison:
        """
        Computes Bonferroni-adjusted confidence intervals and p-values for all pairwise comparisons.

        Parameters:
        alpha : float, optional
            Significance level (default is the class alpha)

        Returns:
        MultinomialComparison
            Standardized results including confidence intervals and p-values
        """
        if alpha is None:
            alpha = self.alpha

        m = self.r * (self.r - 1) // 2
        alpha_adj = alpha / m
        z_alpha = norm.ppf(1 - alpha_adj / 2)

        # Calculate overall test statistic (maximum absolute standardized difference)
        max_z = 0
        comparisons = []
        
        # For z-test, df_error = n - 1 for each comparison
        df_error = self.n - 1
        
        for i in range(self.r):
            for j in range(i + 1, self.r):
                diff = self.p_hat[i] - self.p_hat[j]
                se = np.sqrt((self.p_hat[i] * (1 - self.p_hat[i]) + 
                            self.p_hat[j] * (1 - self.p_hat[j])) / self.n)
                z_stat = abs(diff / se)
                max_z = max(max_z, z_stat)
                
                # Two-sided p-value for this comparison (with Bonferroni adjustment)
                p_value_ij = min(2 * (1 - norm.cdf(z_stat)) * m, 1.0)
                
                ci_lower = diff - z_alpha * se
                ci_upper = diff + z_alpha * se
                comparisons.append((i, j, diff, ci_lower, ci_upper, p_value_ij))

        # Overall p-value (based on maximum test statistic)
        p_value = min(2 * (1 - norm.cdf(max_z)) * m, 1.0)

        return MultinomialComparison(
            method="Bonferroni",
            comparisons=comparisons,
            test_statistic=max_z,
            p_value=p_value,
            alpha=alpha,
            df_num=1,  # Each comparison is a single contrast
            df_error=df_error
        )

    def tukey_confidence_intervals(
            self,
            alpha: Optional[float] = None
        ) -> MultinomialComparison:
        """
        Computes Tukey-adjusted confidence intervals and p-values for all pairwise comparisons.

        Parameters:
        alpha : float, optional
            Significance level (default is the class alpha)

        Returns:
        MultinomialComparison
            Standardized results including confidence intervals and p-values
        """
        if alpha is None:
            alpha = self.alpha

        df = self.n - self.r
        q_alpha = studentized_range.ppf(1 - alpha, self.r, df) / np.sqrt(2)

        # Calculate maximum standardized difference for overall test statistic
        max_q = 0
        comparisons = []
        
        # Degrees of freedom for Tukey
        df_error = self.n - self.r  # Error degrees of freedom
        df_num = self.r - 1         # Numerator degrees of freedom
        
        for i in range(self.r):
            for j in range(i + 1, self.r):
                diff = self.p_hat[i] - self.p_hat[j]
                se = np.sqrt((self.p_hat[i] * (1 - self.p_hat[i]) + 
                            self.p_hat[j] * (1 - self.p_hat[j])) / self.n)
                q_stat = abs(diff / se) * np.sqrt(2)  # Multiply by sqrt(2) for Tukey's Q statistic
                max_q = max(max_q, q_stat)
                
                # Calculate individual comparison p-value
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
            df_num=df_num,
            alpha=alpha,
            df_error=df_error
        )

    
    def _get_comparison_pairs(self, A: np.ndarray) -> List[Tuple[int, int]]:
        """Extracts comparison pairs from constraint matrix rows"""
        pairs = []
        for row in A:
            nonzero = np.nonzero(row)[0]
            if len(nonzero) >= 2:
                pairs.append((nonzero[0], nonzero[1]))
        return pairs
    
    def _create_pairwise_matrix(self) -> np.ndarray:
        """Creates constraint matrix for independent pairwise comparisons.
        Uses a reference category approach where we compare each category
        to the first category."""
        A = np.zeros((self.r - 1, self.r))
        for i in range(1, self.r):
            A[i-1, 0] = 1    # reference category
            A[i-1, i] = -1   # comparison category
        return A
    
def format_test_results(result:MultinomialComparison)-> dict:
    """Format test results into a flat dictionary with confidence interval for the multiple comparisons

    Parameters:
        results : MultinomialComparison
            Test results

    Returns:
        dict: A dictionary with the confidence intervals.
    """

    formatted_data = {
        'test': [],
        'diff': [],
        'p_i-p_j': [],
        'conf_low': [],
        'conf_hi': []
    }

    for comp in result.comparisons:
        i, j = comp[0], comp[1]
        diff = comp[2]
        conf_low, conf_hi = comp[3], comp[4]
        
        formatted_data['test'].append(result.method)
        formatted_data['p_i-p_j'].append(f'{i}-{j}')
        formatted_data['diff'].append(diff)
        formatted_data['conf_low'].append(conf_low)
        formatted_data['conf_hi'].append(conf_hi)

    return formatted_data

def print_test_results(results:MultinomialComparison):
    """
    Print formatted results for multiple comparison tests.

    Parameters:
    results : List[MultinomialComparison]
        List of test results to display
    """
    for result in results:
        print(f"\n{result.method} results:")
        print(f"Overall p-value: {result.p_value:.4f}")
        
        if result.test_statistic is not None:
            print(f"Test statistic: {result.test_statistic:.4f}")
            
        df_str = []
        if result.df_num is not None:
            df_str.append(f"num={result.df_num}")
        if result.df_error is not None:
            df_str.append(f"error={result.df_error}")
        if result.df is not None:
            df_str.append(f"df={result.df}")
        if result.alpha is not None:
            conf = f"{100*(1-result.alpha):.2f}"
            
        if df_str:  # Only print if we have any df information
            print(f"Degrees of freedom: ({', '.join(df_str)})")
            
        print("\nPairwise Comparisons:")
        print(f"Category i\tCategory j\tDifference\t{conf}% CI\t\tp-value")
        print("-" * 80)
        
        for comp in result.comparisons:
            # Handle both 5 and 6-element tuples
            if len(comp) == 6:
                i, j, diff, lower, upper, p_val = comp
            else:
                i, j, diff, lower, upper = comp
                p_val = float('nan')  # Use NaN for missing p-values
                
            print(f"{i}\t\t{j}\t\t{diff:8.3f}\t[{lower:6.3f}, {upper:6.3f}]\t{p_val:.4f}")

