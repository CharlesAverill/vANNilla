import matplotlib.pyplot as plt
from vaNNilla import Random

rand = Random(54)

nums = [rand.next(0, 10) for n in range(10000)]
print(nums)
plt.hist(nums, bins=20, edgecolor='k')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(-1, 11)
plt.show()

