from scipy.stats import chisquare
import numpy as np

x = np.random.uniform(0, 10, size=1000)

hist, _ = np.histogram(x, bins=10)
expected = [len(x)/10] * 10

chi2, p = chisquare(hist, expected)

print(chi2, p)