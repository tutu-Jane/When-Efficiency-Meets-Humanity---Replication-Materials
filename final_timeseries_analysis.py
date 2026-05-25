"""
Supplementary File S6: Reproducible Analysis of Discursive Trajectories
========================================================================
This script reproduces the time-series and structural break analysis 
reported in Section 4.3 of the paper.

Input:
  topic_indices_2000_2025.csv with columns:
    - year: publication year
    - total: total publications per year
    - E_index_share: RCT-aligned discourse share (efficiency-oriented)
    - H_index_share: HFT-aligned discourse share (humanity-oriented)
    - Gap_share: difference (H_index_share - E_index_share)

Outputs:
  1. Piecewise OLS coefficient tables for each candidate breakpoint
  2. Joint F-tests for structural break detection
  3. Quandt Likelihood Ratio (QLR) test as robustness check

Dependencies: pandas, numpy, statsmodels, scipy
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# ============================================================
# Load data
# ============================================================
df = pd.read_csv('/mnt/user-data/uploads/topic_indices_2000_2025.csv')
df = df.sort_values('year').reset_index(drop=True)
df['Year_c'] = df['year'] - df['year'].mean()  # centered year
n = len(df)
print(f"Sample size: n = {n} years ({df['year'].min()} - {df['year'].max()})")
print(f"Centered year reference: mean = {df['year'].mean():.3f}\n")

# ============================================================
# Piecewise OLS model with joint F-test
# ============================================================
def piecewise_ols(df, y_col, break_year):
    """
    Estimate Y = b0 + b1*Year_c + g0*Post + g1*Year_c_Post + e
    where Post = 1 if year >= break_year, 0 otherwise.
    
    Returns full model + joint F-test for H0: g0 = g1 = 0.
    """
    post = (df['year'] >= break_year).astype(int)
    X = sm.add_constant(pd.DataFrame({
        'Year_c': df['Year_c'],
        'Post': post,
        'Year_c_Post': df['Year_c'] * post
    }))
    model = sm.OLS(df[y_col], X).fit()
    
    # Joint F-test: Post = 0 AND Year_c_Post = 0
    f_test = model.f_test('Post = 0, Year_c_Post = 0')
    
    return {
        'model': model,
        'break_year': break_year,
        'F_joint': float(f_test.statistic),
        'p_joint': float(f_test.pvalue),
        'df_num': int(f_test.df_num),
        'df_denom': int(f_test.df_denom),
        'R2': model.rsquared,
        'coefs': model.params.to_dict(),
        'pvals': model.pvalues.to_dict()
    }


# ============================================================
# Run the four pre-specified break tests
# ============================================================
specs = [
    ('H_index_share', 2021, 'HFT (Humanity) discourse, break at 2021'),
    ('E_index_share', 2023, 'RCT (Efficiency) discourse, break at 2023'),
    ('H_index_share', 2023, 'HFT discourse, break at 2023 (robustness)'),
    ('E_index_share', 2021, 'RCT discourse, break at 2021 (robustness)'),
]

print("=" * 75)
print("PRE-SPECIFIED CHOW-TYPE TESTS (Piecewise OLS with joint F-test)")
print("=" * 75)

results_table = []
for y_col, break_year, label in specs:
    r = piecewise_ols(df, y_col, break_year)
    print(f"\n{label}")
    print(f"  Const:        beta = {r['coefs']['const']:.4f}, p = {r['pvals']['const']:.4f}")
    print(f"  Year_c:       beta = {r['coefs']['Year_c']:.4f}, p = {r['pvals']['Year_c']:.4f}")
    print(f"  Post:         beta = {r['coefs']['Post']:.4f}, p = {r['pvals']['Post']:.4f}")
    print(f"  Year_c_Post:  beta = {r['coefs']['Year_c_Post']:.4f}, p = {r['pvals']['Year_c_Post']:.4f}")
    print(f"  R-squared:    {r['R2']:.4f}")
    print(f"  Joint F-test (Post = Year_c_Post = 0): F({r['df_num']},{r['df_denom']}) = {r['F_joint']:.4f}, p = {r['p_joint']:.4f}")
    
    results_table.append({
        'series': y_col,
        'break_year': break_year,
        'F_joint': r['F_joint'],
        'p_joint': r['p_joint'],
        'R2': r['R2'],
        'significant_5pct': r['p_joint'] < 0.05
    })

# ============================================================
# QLR test: scan all candidate years (robustness check)
# ============================================================
print("\n\n" + "=" * 75)
print("QLR (QUANDT LIKELIHOOD RATIO) TEST: SCAN ALL CANDIDATE BREAKPOINTS")
print("=" * 75)
print("Andrews (1993) critical values for q=2 restrictions, 15% trim:")
print("  1% = 15.37,  5% = 11.79,  10% = 9.84")

# 15% trim region
trim = 0.15
lower_idx = int(np.ceil(trim * n))
upper_idx = int(np.floor((1 - trim) * n))

def qlr_scan(df, y_col):
    f_stats = []
    for break_year in df['year'].iloc[lower_idx:upper_idx]:
        r = piecewise_ols(df, y_col, break_year)
        f_stats.append({'year': int(break_year), 'F': r['F_joint'], 'p': r['p_joint']})
    return pd.DataFrame(f_stats)

for y_col, label in [('H_index_share', 'HFT discourse'), 
                      ('E_index_share', 'RCT discourse'),
                      ('Gap_share', 'Gap (HFT - RCT)')]:
    scan = qlr_scan(df, y_col)
    sup_row = scan.loc[scan['F'].idxmax()]
    print(f"\n{label}:")
    print(f"  Trim region: {df['year'].iloc[lower_idx]} to {df['year'].iloc[upper_idx-1]}")
    print(f"  sup F = {sup_row['F']:.4f} at year {int(sup_row['year'])}")
    
    if sup_row['F'] > 15.37:
        sig = "p < 0.01"
    elif sup_row['F'] > 11.79:
        sig = "p < 0.05"
    elif sup_row['F'] > 9.84:
        sig = "p < 0.10"
    else:
        sig = "n.s. (does not reject H0 of no break)"
    print(f"  QLR significance: {sig}")

# ============================================================
# Pre/Post 2021 mean comparison (as descriptive evidence)
# ============================================================
print("\n\n" + "=" * 75)
print("PRE/POST 2021 MEAN COMPARISON (descriptive evidence)")
print("=" * 75)
for y_col, label in [('H_index_share', 'HFT discourse share'), 
                      ('E_index_share', 'RCT discourse share')]:
    pre = df.loc[df['year'] < 2021, y_col]
    post = df.loc[df['year'] >= 2021, y_col]
    t_stat, p_val = stats.ttest_ind(post, pre, equal_var=False)  # Welch's t-test
    u_stat, u_p = stats.mannwhitneyu(post, pre, alternative='two-sided')
    print(f"\n{label}:")
    print(f"  Pre-2021 mean (n={len(pre)}):  {pre.mean():.4f} (SD = {pre.std():.4f})")
    print(f"  Post-2021 mean (n={len(post)}): {post.mean():.4f} (SD = {post.std():.4f})")
    print(f"  Welch's t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"  Mann-Whitney U: U = {u_stat:.1f}, p = {u_p:.4f}")

# ============================================================
# Save summary table
# ============================================================
summary = pd.DataFrame(results_table)
summary.to_excel('/mnt/user-data/outputs/Time_Series_Analysis_Supplementary_S6.xlsx', 
                  index=False, sheet_name='ChowTest_Results')
print("\n\nResults saved to /mnt/user-data/outputs/Time_Series_Analysis_Supplementary_S6.xlsx")
