import numpy as np
import pandas as pd

# ---------- helper functions (replace math.erf / math.gamma) ----------
def erf(x):
    # Abramowitz-Stegun approximation of the error function
    sign = np.sign(x)
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y

def norm_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2)))

def log_gamma(z):
    # Lanczos approximation of ln(Gamma(z))
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    z = z - 1
    x = c[0]
    for i in range(1, 9):
        x += c[i] / (z + i)
    t = z + 7.5
    return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)


# ==========================================
# PART 1 : PEARSON CORRELATION ANALYSIS
# ==========================================
print("PART 1: PEARSON CORRELATION ANALYSIS")

np.random.seed(42)
n = 50
hours_studied = np.random.uniform(1, 10, n)
exam_score = 50 + (hours_studied * 4.5) + np.random.normal(0, 5, n)

# Pearson r
m_h, m_s = hours_studied.mean(), exam_score.mean()
num = np.sum((hours_studied - m_h) * (exam_score - m_s))
den = np.sqrt(np.sum((hours_studied - m_h) ** 2) * np.sum((exam_score - m_s) ** 2))
r_stat = num / den if den != 0 else 0.0

# t-statistic and p-value (normal approximation)
df_pearson = n - 2
t_stat = r_stat * np.sqrt(df_pearson / (1 - r_stat ** 2)) if abs(r_stat) < 1.0 else float('inf')
p_value_pearson = 2.0 * (1.0 - norm_cdf(abs(t_stat)))

print(f"Pearson Correlation Coefficient (r): {r_stat:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value_pearson:.4e}")

if p_value_pearson < 0.05:
    print("Result: Statistically significant linear relationship (p < 0.05).\n")
else:
    print("Result: No statistically significant linear relationship (p >= 0.05).\n")

# Correlation matrix using pandas
df = pd.DataFrame({'Hours_Studied': hours_studied, 'Exam_Score': exam_score})
print("Correlation Matrix:")
print(df.corr().round(4))


# ==========================================
# PART 2 : CHI-SQUARE TEST OF INDEPENDENCE
# ==========================================
print("\nPART 2: CHI-SQUARE TEST OF INDEPENDENCE")

np.random.seed(42)
edu_levels = ['High School', 'Bachelor', 'Master']
job_sats = ['Low', 'Medium', 'High']
n_samples = 150

edu_sample = np.random.choice(edu_levels, size=n_samples, p=[0.3, 0.5, 0.2])
sat_sample = np.random.choice(job_sats, size=n_samples, p=[0.25, 0.45, 0.3])

# Observed contingency table
observed_df = pd.crosstab(pd.Series(edu_sample, name='Education'),
                          pd.Series(sat_sample, name='Satisfaction'))
observed_df = observed_df.reindex(index=edu_levels, columns=job_sats, fill_value=0)

print("Observed Contingency Table:")
print(observed_df.assign(Total=observed_df.sum(axis=1))
      .pipe(lambda d: pd.concat([d, d.sum().rename('Total').to_frame().T])))

# Expected frequencies and chi-square statistic
observed = observed_df.values.astype(float)
row_totals = observed.sum(axis=1)
col_totals = observed.sum(axis=0)
expected = np.outer(row_totals, col_totals) / n_samples

chi2_stat = np.sum((observed - expected) ** 2 / expected)
dof = (len(edu_levels) - 1) * (len(job_sats) - 1)

# p-value from chi-square survival function (regularized lower incomplete gamma)
s = dof / 2.0
x_half = chi2_stat / 2.0
term = sum_term = 1.0 / s
for k in range(1, 200):
    term *= x_half / (s + k)
    sum_term += term
    if term < 1e-12 * sum_term:
        break
lower_reg = np.exp(s * np.log(x_half) - x_half - log_gamma(s)) * sum_term
p_val_chi2 = max(0.0, min(1.0, 1.0 - lower_reg))

print(f"\nChi-Square Statistic (chi2): {chi2_stat:.4f}")
print(f"Degrees of Freedom (dof): {dof}")
print(f"p-value: {p_val_chi2:.4f}")

print("\nExpected Frequencies Matrix:")
print(pd.DataFrame(expected, index=edu_levels, columns=job_sats).round(2))

print("\nInterpretation:")
if p_val_chi2 < 0.05:
    print("Reject H0: There is a statistically significant association between the two categorical variables.")
else:
    print("Fail to Reject H0: No significant association found (they are independent).")