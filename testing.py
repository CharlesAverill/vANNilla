import matplotlib.pyplot as plt
from vaNNilla import rand_range

rand = rand_range(0, 10)

nums = [next(rand) for n in range(10000)]
print(nums)
plt.hist(nums, bins=20, edgecolor='k')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(-1, 11)
plt.show()

