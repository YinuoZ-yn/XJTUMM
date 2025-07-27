import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from matplotlib.font_manager import FontProperties
plt.rcParams['figure.facecolor']='#F4FCF8'
plt.rcParams['axes.facecolor']='#F4FCF8'
# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
excel_file_1 = pd.ExcelFile('国赛校赛第一问数据2.0.xlsx')
sheet_names_1 = excel_file_1.sheet_names

excel_file_2 = pd.ExcelFile('预测GDP入学.xlsx')
sheet_names_2 = excel_file_2.sheet_names

excel_file_3 = pd.ExcelFile('灵敏度1.xlsx')
sheet_names_3 = excel_file_2.sheet_names

excel_file_4 = pd.ExcelFile('灵敏度1.2.xlsx')
sheet_names_4 = excel_file_2.sheet_names

df1 = excel_file_1.parse('Sheet1')
df2 = excel_file_2.parse('Sheet1')
df3 = excel_file_3.parse('Sheet1')
df4 = excel_file_4.parse('Sheet1')
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df1[['GDP（亿元）', '高等教育毛入学率（%）']])

model = LinearRegression()
model.fit(X_poly, df1['人口总数（万人）'])

future_X_poly = poly.transform(df2[['GDP（亿元）', '高等教育毛入学率（%）']])

future_X_poly01 = poly.transform(df3[['GDP（亿元）', '高等教育毛入学率（%）']])
future_X_poly02 = poly.transform(df4[['GDP（亿元）', '高等教育毛入学率（%）']])

future_population = model.predict(future_X_poly)
future_population01 = model.predict(future_X_poly01)
future_population02 = model.predict(future_X_poly02)

df2['人口总数（万人）预测值'] = future_population

df3['人口总数（万人）预测值（小幅扰动毛入学率）'] = future_population01
df4['人口总数（万人）预测值（小幅扰动GDP）'] = future_population02

y_pred = model.predict(X_poly)

residuals = df1['人口总数（万人）'] - y_pred
mse = mean_squared_error(df1['人口总数（万人）'], y_pred)
r2 = r2_score(df1['人口总数（万人）'], y_pred)
# 计算置信区间
n = len(df1)
p = X_poly.shape[1]
dof = n - p - 1
t = stats.t.ppf(0.975, dof)
residuals = df1['人口总数（万人）'] - y_pred
std_error = np.sqrt(np.sum(residuals**2) / dof)
margin_of_error = t * std_error * np.sqrt(1 + np.diag(X_poly @ np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T))
lower_bound = y_pred - margin_of_error
upper_bound = y_pred + margin_of_error

XtX_inv = np.linalg.inv(X_poly.T @ X_poly)
se_future = np.sqrt(mse * (1 + np.diag(future_X_poly @ XtX_inv @ future_X_poly.T)))
lower_bound_future = future_population - t * se_future
upper_bound_future = future_population + t * se_future

# 绘制预测结果和置信区间
plt.figure(figsize=(12, 6))
plt.plot(df1['年份'], df1['人口总数（万人）'], color='#ff6f91',label='实际人口总数', marker='o')
plt.plot(df2['年份'], df2['人口总数（万人）预测值'],color='#FAAB52', label='未来人口总数预测值', marker='o')
plt.fill_between(df1['年份'], lower_bound, upper_bound, color='#B694ED', alpha=0.2, label='95% 置信区间')
plt.fill_between(df2['年份'], lower_bound_future, upper_bound_future, color='lightgreen', alpha=0.2, label='预测数据95%置信区间')

plt.plot(df2['年份'], df3['人口总数（万人）预测值（小幅扰动毛入学率）'],color='#b0d198', label='未来人口总数预测值（小幅扰动毛入学率）', marker='o')
plt.plot(df2['年份'], df4['人口总数（万人）预测值（小幅扰动GDP）'],color='#74b193', label='未来人口总数预测值（小幅扰动GDP）', marker='o')

# 标注2030年，2035年和2045年的数据
for year in [2030, 2035, 2045]:
    if year in df2['年份'].values:
        pop = df2[df2['年份'] == year]['人口总数（万人）预测值'].values[0]
        plt.annotate(f'{year}年: {pop:.2f}万人', xy=(year, pop), xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle='->'))
plt.xticks(np.arange(min(df1['年份'].min(), df2['年份'].min()), max(df1['年份'].max(), df2['年份'].max()) + 1, 5))
plt.xlabel('年份',fontsize=15)
plt.ylabel('人口总数（万人）',fontsize=15)
plt.title('人口总数预测及置信区间')
plt.legend()
plt.grid(axis='y',linestyle='--')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()