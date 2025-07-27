import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
file_path = "第三问整合1.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")


df['GDP增长率扰动(%)'] = ((df['GDP增长指数'] - df.loc[0, 'GDP增长指数']) / df.loc[0, 'GDP增长指数']) * 100

# 2. 计算研究生毕业人数增长率 S_t（供给增速差扰动）
df['研究生毕业人数增长率(%)'] = df['研究生毕业人数'].pct_change() * 100
df['研究生毕业人数增长率(%)'].fillna(0, inplace=True)  # 第一年填充为0
cmap = mcolors.LinearSegmentedColormap.from_list('bright', ['#e2ff53', '#7ae376', '#00bf91'])
# 3. 标准化气泡大小（工资比，列名是'高/平均'）
bubble_size = (df['高/平均'] * 100) ** 2/8  # 放大差异

# 绘制气泡图
plt.figure(figsize=(15, 10))
scatter = plt.scatter(
    x=df['GDP增长率扰动(%)'],
    y=df['研究生毕业人数增长率(%)'],
    s=bubble_size,
    c=df['失业率'],
    cmap=cmap,
    alpha=0.7,
    edgecolors='black'
)

# 添加标签和标题
plt.xlabel('GDP增长率扰动幅度 (%)', fontsize=12)
plt.ylabel('研究生毕业人数增长率 (%)', fontsize=12)
plt.title('双自变量交互气泡图', fontsize=14)

# 颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('失业率 (%)', fontsize=12)

legend_sizes = [1.0, 1.5, 2.0]  # 工资比示例值
legend_labels = [f'工资比={size}' for size in legend_sizes]
legend_bubble_sizes = [(size * 30) ** 2 for size in legend_sizes]

# 创建横向图例的句柄
legend_handles = [
    plt.scatter([], [], s=size, color='#e2ff53', alpha=0.6, edgecolors='black', label=label)
    for size, label in zip(legend_bubble_sizes, legend_labels)
]

# 添加横向图例（位于底部）
plt.legend(
    handles=legend_handles,
    loc='lower center',          # 定位到底部中心
    bbox_to_anchor=(0.2, 0.9), # 调整图例位置（向下偏移）
    ncol=len(legend_sizes),      # 横向排列（列数=示例数量）
    frameon=True,
    borderpad=2.5,
    framealpha=0.8,
    scatterpoints=1,             # 每个图例显示一个气泡
    fontsize=10,
    title_fontsize=12
)


target_years = [1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010, 2011, 2012,2013,2014,2017,2018,2019,2020,2021,2022,2023]  # 替换为实际需要标记的年份
for i, year in enumerate(df['年份']):
    if year in target_years:  # 只对目标年份添加注释
        plt.annotate(
            year,
            xy=(df['GDP增长率扰动(%)'][i], df['研究生毕业人数增长率(%)'][i]),
            xytext=(1, 1),  # 文本偏移量（避免重叠）
            textcoords='offset points',
            fontsize=10,
            ha='center',
        )

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()