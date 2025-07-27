import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
import numpy as np
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)
plt.rcParams['font.family'] = font.get_name()
plt.rcParams['axes.unicode_minus'] = False
# 读取文件
excel_file = pd.ExcelFile('第三问整合1.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')
# 筛选 2010 - 2023 年的数据
selected_df = df[(df['年份'] >= 1992) & (df['年份'] <= 2023)]



x_new = np.linspace(selected_df['年份'].min(), selected_df['年份'].max(), 500)

# 对标准化1进行插值
f1 = interp1d(selected_df['年份'], selected_df['标准化1'], kind='cubic')
y1_smooth = f1(x_new)

# 对标准化2进行插值
f2 = interp1d(selected_df['年份'], selected_df['标准化5'], kind='cubic')
y2_smooth = f2(x_new)

# 对标准化3进行插值，处理可能存在的缺失值
f3 = interp1d(selected_df['年份'], selected_df['标准化3'], kind='cubic')
y3_smooth = f3(x_new)

# 绘制平滑折线图
plt.figure(figsize=(10, 6))
plt.plot(x_new, y1_smooth, label='第三产业占GDP比重（D）',color='#edfc78')
plt.plot(x_new, y3_smooth, label='Y1（本科以上毕业生平均工资 / 社会平均工资）',color='#90b6a5')
plt.plot(x_new, y2_smooth, label='本科毕业生人数（S）',color='#bea887')


plt.scatter(selected_df['年份'], selected_df['标准化1'], marker='o', color='#edfc78')
plt.scatter(selected_df['年份'], selected_df['标准化5'], marker='s', color='#bea887')
plt.scatter(selected_df['年份'], selected_df['标准化3'], marker='^', color='#90b6a5')
# 添加标题和标签
plt.title('1992 - 2023 年 3 个标准化值折线图')
plt.xlabel('年份',fontsize='12')
plt.ylabel('标准化值',fontsize='12')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
# 添加图例
plt.legend()
# 显示网格
plt.grid(axis='y',linestyle='--')

# 显示图形
plt.show()