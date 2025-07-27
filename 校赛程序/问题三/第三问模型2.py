import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
# 1. 数据读取与预处理
df = pd.read_excel('第三问整合1.xlsx')
df = df[df['年份']>=2018]
# 提取需要的列
data = df[['年份', '失业率', '研究生毕业人数', 'GDP增长指数']].copy()

# 计算研究生毕业人数增长率(S_t)
data['S_t'] = data['研究生毕业人数'].pct_change() * 100  # 转换为百分比

# 计算GDP增长率(G_t)，假设GDP增长指数基期为100
data['G_t'] = data['GDP增长指数'].pct_change() * 100

# 计算供给增速差(S_t - G_t)
data['供给增速差'] = data['S_t'] - data['G_t']

# 删除缺失值（第一年没有增长率）
data = data.dropna()

# 2. 多元线性回归模型构建
X = data[['供给增速差', 'G_t']]
y = data['失业率']
X = sm.add_constant(X)  # 添加常数项

model = sm.OLS(y, X)
results = model.fit()

# 3. 结果分析
print("="*60)
print("供需失衡与就业贬值模型回归结果")
print("="*60)
print(results.summary())
print("\n关键结论：")
print(f"- 供给增速差系数: {results.params[1]:.4f} (p值={results.pvalues[1]:.4f})")
print(f"- GDP增长率系数: {results.params[2]:.4f} (p值={results.pvalues[2]:.4f})")
print(f"- 模型R平方: {results.rsquared:.4f}")

# 4. 可视化分析
plt.figure(figsize=(18, 6))

# 子图1：供给增速差与失业率关系
plt.subplot(1, 3, 1)
plt.scatter(data['供给增速差'], y, alpha=0.7)
plt.xlabel('供给增速差 (S_t - G_t)%')
plt.ylabel('失业率(%)')
plt.title('供给过剩对失业率的影响')
plt.grid(True)

# 添加回归线（控制其他变量）
x_pred = np.linspace(data['供给增速差'].min(), data['供给增速差'].max(), 100)
g_mean = data['G_t'].mean()  # 固定GDP增长率为均值
y_pred = results.params[0] + results.params[1]*x_pred + results.params[2]*g_mean
plt.plot(x_pred, y_pred, 'r-')

# 子图2：GDP增长率与失业率关系
plt.subplot(1, 3, 2)
plt.scatter(data['G_t'], y, alpha=0.7)
plt.xlabel('GDP增长率(%)')
plt.ylabel('失业率(%)')
plt.title('经济增长对失业率的影响')
plt.grid(True)

# 添加回归线（控制其他变量）
x_pred = np.linspace(data['G_t'].min(), data['G_t'].max(), 100)
s_mean = data['供给增速差'].mean()  # 固定供给增速差为均值
y_pred = results.params[0] + results.params[1]*s_mean + results.params[2]*x_pred
plt.plot(x_pred, y_pred, 'r-')

# 子图3：3D关系图
ax = plt.subplot(1, 3, 3, projection='3d')
ax.scatter(data['供给增速差'], data['G_t'], y, c='b', marker='o')
ax.set_xlabel('供给增速差(%)')
ax.set_ylabel('GDP增长率(%)')
ax.set_zlabel('失业率(%)')
ax.set_title('三维影响因素关系')

# 生成回归平面
x_surf = np.linspace(data['供给增速差'].min(), data['供给增速差'].max(), 10)
y_surf = np.linspace(data['G_t'].min(), data['G_t'].max(), 10)
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
z_pred = results.params[0] + results.params[1]*x_surf + results.params[2]*y_surf
ax.plot_surface(x_surf, y_surf, z_pred, color='r', alpha=0.3)

plt.tight_layout()
plt.show()

# 5. 输出关键指标表格
result_table = pd.DataFrame({
    '年份': data['年份'],
    '失业率(%)': data['失业率'],
    '研究生毕业人数增长率(%)': data['S_t'].round(2),
    'GDP增长率(%)': data['G_t'].round(2),
    '供给增速差(%)': data['供给增速差'].round(2),
    '模型预测失业率(%)': results.fittedvalues.round(2)
})

print("\n详细计算结果：")
print(result_table.to_string(index=False))