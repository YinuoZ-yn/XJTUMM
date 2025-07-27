import pandas as pd
import numpy as np

# 读取文件
excel_file = pd.ExcelFile('第二问指标.xlsx')

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')

# 查看数据的基本信息
print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan'))
# 提取需要的指标数据
selected_columns = ['研究生毕业生占比', '高等学校专任教师数占比', '师生比', '生均教育经费']
new_df = df[selected_columns]

# 对数据进行标准化
P = new_df.div(new_df.sum(axis=0), axis=1)

# 计算每个指标的熵值
E = -(P * np.log(P).replace([np.inf, -np.inf], 0)).sum(axis=0) / np.log(len(P))

# 计算每个指标的差异系数
G = 1 - E

# 计算每个指标的权重
W = G / G.sum()

# 输出结果
result = pd.DataFrame({
    '指标': selected_columns,
    '权重': W
})

print(result)