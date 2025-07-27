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
for sheet_name in sheet_names_1:
    df = excel_file_1.parse(sheet_name)
    #print(f'sheet表名为{sheet_name}的基本信息：')
    df.info()
    rows, columns = df.shape
    # if rows < 100 and columns < 20:
    #     #print(f'sheet表名为{sheet_name}的全部内容信息：')
    #     #print(df.to_csv(sep='\t', na_rep='nan'))
    # else:
        # 长表数据查看数据前几行信息
        #print(f'sheet表名为{sheet_name}的前几行内容信息：')
        #print(df.head().to_csv(sep='\t', na_rep='nan'))

excel_file_2 = pd.ExcelFile('预测GDP入学.xlsx')
sheet_names_2 = excel_file_2.sheet_names
for sheet_name in sheet_names_2:
    df = excel_file_2.parse(sheet_name)
    #print(f'sheet表名为{sheet_name}的基本信息：')
    df.info()

    rows, columns = df.shape

    # if rows < 100 and columns < 20:
    #
    #     print(f'sheet表名为{sheet_name}的全部内容信息：')
    #     print(df.to_csv(sep='\t', na_rep='nan'))
    # else:
    #
    #     print(f'sheet表名为{sheet_name}的前几行内容信息：')
    #     print(df.head().to_csv(sep='\t', na_rep='nan'))

df1 = excel_file_1.parse('Sheet1')
df2 = excel_file_2.parse('Sheet1')
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(df1[['GDP（亿元）', '高等教育毛入学率（%）']])

model = LinearRegression()
model.fit(X_poly, df1['人口总数（万人）'])

future_X_poly = poly.transform(df2[['GDP（亿元）', '高等教育毛入学率（%）']])

future_population = model.predict(future_X_poly)

df2['人口总数（万人）预测值'] = future_population

y_pred = model.predict(X_poly)

residuals = df1['人口总数（万人）'] - y_pred

# 绘制残差图
plt.figure(figsize=(10, 6))
for x, y in zip(y_pred, residuals):
    half_length = abs(y) *5
    plt.vlines(x, y - half_length, y + half_length, color='#D8BFD8', alpha=0.95)
plt.scatter(y_pred, residuals,marker='o',color= '#FFB6C1')
plt.axhline(y=0, color='#FFA500', linestyle='--')
plt.xlabel('预测值（万人）',fontsize=15)
#plt.xticks(rotation=45)
plt.ylabel('残差（万人）',fontsize=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('人口总数预测残差图')
plt.grid(axis='y',linestyle='--')
plt.show()

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

# 生成网格数据用于绘制三维曲面
gdp_min, gdp_max = df1['GDP（亿元）'].min(), df1['GDP（亿元）'].max()
enrollment_min, enrollment_max = df1['高等教育毛入学率（%）'].min(), df1['高等教育毛入学率（%）'].max()
gdp_grid, enrollment_grid = np.meshgrid(
    np.linspace(gdp_min, gdp_max, 100),
    np.linspace(enrollment_min, enrollment_max, 100)
)

# 将网格数据转换为适合模型预测的格式
grid_data = np.column_stack([gdp_grid.ravel(), enrollment_grid.ravel()])
grid_poly = poly.transform(grid_data)

# 进行预测
predicted_population = model.predict(grid_poly)
predicted_population = predicted_population.reshape(gdp_grid.shape)

# 创建三维图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维曲面
ax.plot_surface(gdp_grid, enrollment_grid, predicted_population, cmap='viridis', alpha=0.5)
ax.scatter(df1['GDP（亿元）'], df1['高等教育毛入学率（%）'], df1['人口总数（万人）'], c='r', marker='o')
# 设置坐标轴标签和标题
ax.set_xlabel('GDP（亿元）')
ax.set_ylabel('高等教育毛入学率（%）')
ax.set_zlabel('人口总数（万人）')
ax.set_title('多元非线性回归模型预测人口总数三维曲面')

plt.show()

print('模型均方误差：', mse)
print('模型决定系数：', r2)
print('未来 20 年人口总数预测结果：')
print(df2[['年份', '人口总数（万人）预测值']])