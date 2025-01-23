import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 生成示例数据
np.random.seed(42)
data1 = np.load('/data/LiuYuyao/Project/WinVL/checkpoint/temp_vectorVIS/epoch15.npy')
data1 = np.transpose(data1)
data2 = np.load('/data/LiuYuyao/Project/WinVL/checkpoint/temp_vectorVIS/epoch15_constant.npy')

# 对 data1 和 data2 进行裁剪
data1 = np.clip(data1, -0.1, 0.1)
data2 = np.clip(data2, -0.1, 0.1)

# 对 data2 的坐标值取反
data2_neg = -1 * data2

# 应用 KMeans 聚类
kmeans1 = KMeans(n_clusters=2, random_state=42)
labels1 = kmeans1.fit_predict(data1)

kmeans2 = KMeans(n_clusters=2, random_state=1)
labels2 = kmeans2.fit_predict(data2_neg)

# 绘制聚类后的数据
plt.figure(figsize=(14,8))

from matplotlib.colors import ListedColormap
cmap_data1 = ListedColormap(['red', 'blue'])
cmap_data2 = ListedColormap(['grey', 'yellow'])

# 绘制 data1

# 绘制 data2（坐标值取反）
scatter1 = plt.scatter(data1[:, 0],-1 * data1[:, 1], c=labels1, cmap=cmap_data1, s=50, alpha=0.6, marker='*', label='Learnable Prompt')
scatter2 = plt.scatter(data2_neg[:, 0],  data2_neg[:, 1], c=labels2, cmap=cmap_data2, s=50, alpha=0.3, label='Template Prompt')

# 添加图例
plt.legend()

# 设置标题和标签
# plt.title("Clustering Visualization with KMeans (2 Clusters)")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")

# 设置坐标轴数值格式
import matplotlib.ticker as ticker
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

ax.tick_params(axis='both', which='major', labelsize=16)

# 添加颜色条
cbar1 = plt.colorbar(scatter2, extend='both')
cbar1.set_label('Learnable Prompt', fontsize=16)
cbar1.ax.tick_params(labelsize=16)  # 调整颜色条刻度标签的字体大小

cbar2 = plt.colorbar(scatter1, extend='both')
cbar2.set_label('Template Prompt', fontsize=16)
cbar2.ax.tick_params(labelsize=16)  # 调整颜色条刻度标签的字体大小

plt.show()