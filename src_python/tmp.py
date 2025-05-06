import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm

a, b = 0, 21

loc, scale = 1.3, 10.5

a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale

rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
x = np.linspace(truncnorm.ppf(0.01, a, b), truncnorm.ppf(1, a, b), 100)
r = rv.rvs(size=1)

print(r)

fig, ax = plt.subplots(1, 1)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim(a, b)
ax.legend(loc='best', frameon=False)

fig.savefig("clipped_normal_dist.pdf")