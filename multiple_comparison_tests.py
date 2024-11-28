# multiple_comparisons_tests.py
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.stats import chi2, norm, studentized_range

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
        A: np.ndarray,
        alpha: Optional[float] = None,
        bonferroni_correction = False,
        fisher_correction: bool = False
    ) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        """
        Performs the Scheffé-Wald test for a multinomial distribution and computes confidence intervals.

        Parameters:
        A : 2D array-like
            Constraint matrix A (dimensions u x r)
        alpha : float, optional
            Significance level for confidence intervals (default is the class alpha)
        fisher_correction : bool, optional
            If True, applies the Fisher correction to the test statistic (default is False)
        bonferroni_correction : bool, optional
            If True, adjust alpha using Bonferroni correction.

        Returns:
        W : float
            The test statistic
        p_value : float
            The p-value for the test
        df : int
            Degrees of freedom (u)
        ci_lower : array-like
            Lower bounds of the confidence intervals
        ci_upper : array-like
            Upper bounds of the confidence intervals
        """
        
        if alpha is None:
            alpha = self.alpha

        n = self.n
        p_hat = self.p_hat
        r = self.r
        u = A.shape[0]  # Number of constraints

        # Dimension check for A
        if A.shape[1] != r:
            raise ValueError(f"Constraint matrix A must have {r} columns, matching the number of categories.")

        # Compute the covariance matrix Σ_p_hat
        diag_p_hat = np.diag(p_hat)
        outer_p_hat = np.outer(p_hat, p_hat)
        Sigma_p_hat = (diag_p_hat - outer_p_hat)/n # Covariance matrix

        # Compute W = p_hat' A' (A Σ_p_hat A')^{-1} A p_hat
        A = np.array(A)
        Ap = A @ p_hat  # A * p_hat

        A_Sigma_A_T = A @ Sigma_p_hat @ A.T

        # Check if A_Sigma_A_T is invertible
        if np.linalg.matrix_rank(A_Sigma_A_T) < u:
            raise np.linalg.LinAlgError("Matrix A Σ_p_hat A' is singular and cannot be inverted.")

        # Compute inverse
        inv_A_Sigma_A_T = np.linalg.inv(A_Sigma_A_T)

        W = Ap.T @ inv_A_Sigma_A_T @ Ap  # Wald Test statistic

        # Compute standard errors
        se = np.sqrt(np.diag(A_Sigma_A_T))

        if fisher_correction:
            # Apply Fisher correction for small sample sizes
            correction_factor = (n - 1) / n
            W *= correction_factor
            se *= np.sqrt(correction_factor)

        # Degrees of freedom
        df = u

        # Compute p-value
        p_value = 1 - chi2.cdf(W, df)

        if bonferroni_correction:
            m = u
            alpha_adj = alpha / m
        else:
            alpha_adj = alpha

        # Compute critical value for confidence intervals using the Scheffé adjustment
        critical_value = chi2.ppf(1 - alpha_adj, df)
            
        # Confidence intervals
        ci_lower = Ap - np.sqrt(critical_value) * se
        ci_upper = Ap + np.sqrt(critical_value) * se

        return W, p_value, df, ci_lower, ci_upper

    def bonferroni_confidence_intervals(
        self,
        alpha: Optional[float] = None
    ) -> List[Tuple[int, int, float, float, float]]:
        """
        Computes Bonferroni-adjusted confidence intervals for all pairwise comparisons.

        Parameters:
        alpha : float, optional
            Significance level for confidence intervals (default is the class alpha)

        Returns:
        comparisons : list of tuples
            Each tuple contains (i, j, diff, ci_lower, ci_upper)
        """
        if alpha is None:
            alpha = self.alpha

        n = self.n
        p_hat = self.p_hat
        r = self.r

        # Number of pairwise comparisons
        m = r * (r - 1) / 2

        # Adjusted alpha
        alpha_adj = alpha / m

        # Critical value
        z_alpha = norm.ppf(1 - alpha_adj / 2)

        comparisons = []
        for i in range(r):
            for j in range(i + 1, r):
                diff = p_hat[i] - p_hat[j]
                se = np.sqrt((p_hat[i] * (1 - p_hat[i]) + p_hat[j] * (1 - p_hat[j])) / n)
                ci_lower = diff - z_alpha * se
                ci_upper = diff + z_alpha * se
                comparisons.append((i, j, diff, ci_lower, ci_upper))
        return comparisons

    def tukey_confidence_intervals(
        self,
        alpha: Optional[float] = None
    ) -> List[Tuple[int, int, float, float, float]]:
        """
        Computes Tukey-adjusted confidence intervals for all pairwise comparisons.

        Parameters:
        alpha : float, optional
            Significance level for confidence intervals (default is the class alpha)

        Returns:
        comparisons : list of tuples
            Each tuple contains (i, j, diff, ci_lower, ci_upper)
        """
        if alpha is None:
            alpha = self.alpha

        n = self.n
        p_hat = self.p_hat
        r = self.r

        # Degrees of freedom for Tukey's method (approximation)
        df = n - r

        # Critical value
        q_alpha = studentized_range.ppf(1 - alpha, r, df) / np.sqrt(2)

        comparisons = []
        for i in range(r):
            for j in range(i + 1, r):
                diff = p_hat[i] - p_hat[j]
                se = np.sqrt((p_hat[i] * (1 - p_hat[i]) + p_hat[j] * (1 - p_hat[j])) / n)
                ci_lower = diff - q_alpha * se
                ci_upper = diff + q_alpha * se
                comparisons.append((i, j, diff, ci_lower, ci_upper))
        return comparisons
