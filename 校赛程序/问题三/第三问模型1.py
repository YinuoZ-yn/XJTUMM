import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
# 1. 数据读取与预处理
df = pd.read_excel('第三问整合1.xlsx')
df = df[df['年份']>=2013]
# 提取需要的列
data = df[['年份', '高/平均', '研究生毕业人数', 'GDP增长指数']].copy()

# 计算研究生毕业人数增长率(S_t)
data['S_t'] = data['研究生毕业人数'].pct_change() * 100  # 转换为百分比

# 计算GDP增长率(G_t)，假设GDP增长指数基期为100
data['G_t'] = data['GDP增长指数'].pct_change() * 100

# 计算供给增速差(S_t - G_t)
data['供给增速差'] = data['S_t'] - data['G_t']

# 删除缺失值（第一年没有增长率）
data = data.dropna()

# 2. 模型构建
X = data['供给增速差']
y = data['高/平均']
X = sm.add_constant(X)  # 添加常数项

model = sm.OLS(y, X)
results = model.fit()

print("="*50)
print("供需失衡与收入贬值模型回归结果")
print("="*50)
print(results.summary())
print("\n关键结论：")
print(f"- 供给增速差系数: {results.params[1]:.4f} (p值={results.pvalues[1]:.4f})")
print(f"- 模型R平方: {results.rsquared:.4f}")

# 4. 可视化
plt.figure(figsize=(12, 6))



# 散点图与回归线
plt.subplot(1, 2, 1)
plt.scatter(data['供给增速差'], y, alpha=0.7,color='#949fff')
plt.plot(data['供给增速差'], results.fittedvalues, 'r-',color='#ffe8ff')
plt.xlabel('供给增速差 (S_t - G_t)%')
plt.ylabel('高学历/平均工资比值')
plt.title('供需关系对工资比值的影响')
plt.grid(axis='y',linestyle='--')
plt.legend()
# 残差诊断图
plt.subplot(1, 2, 2)
plt.scatter(results.fittedvalues, results.resid, alpha=0.7,color='#ffba8c')
plt.axhline(y=0, color='#b3a7b7', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差诊断图')
plt.grid(axis='y',linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
# 5. 输出关键指标表格
result_table = pd.DataFrame({
    '年份': data['年份'],
    '高学历/平均工资': data['高/平均'],
    '研究生毕业人数增长率(%)': data['S_t'].round(2),
    'GDP增长率(%)': data['G_t'].round(2),
    '供给增速差(%)': data['供给增速差'].round(2),
    '模型预测值': results.fittedvalues.round(4)
})

print("\n详细计算结果：")
print(result_table.to_string(index=False))