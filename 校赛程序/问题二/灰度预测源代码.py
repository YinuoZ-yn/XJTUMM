import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def grey_model(data):
    AGO = np.cumsum(data)

    Z = (AGO[:-1] + AGO[1:]) / 2.0

    # 构造矩阵B和Y
    B = np.vstack((-Z, np.ones(len(Z)))).T
    Y = data[1:].reshape(len(Z), 1)

    # 计算参数a和b
    a, b = np.dot(np.linalg.pinv(B), Y)

    # 预测函数
    def predict(k):
        return (data[0] - b / a) * np.exp(-a * k) + b / a

    # 计算模拟值
    sim_values = []
    sim_values.append(data[0])
    for i in range(1, len(data)):
        sim_values.append(predict(i))

    # 模型检验
    errors = data - sim_values
    S1 = np.std(data, ddof=1)
    S2 = np.std(errors, ddof=1)
    C = S2 / S1
    P = np.sum(np.abs(errors - np.mean(errors)) < 0.6745 * S1) / len(data)

    return predict, sim_values, (a[0], b[0]), (C[0], P[0])


# 1. 读取历史数据
history_df = pd.read_excel('生均教育经费.xlsx')
years = history_df['年份'].values
population = history_df['人口总数（万人）'].values
expenditure = history_df['生均教育经费(元)'].values

# 2. 灰度模型预测
predict_func, sim_values, (a, b), (C, P) = grey_model(expenditure)

print(f"灰色模型参数: a={a:.4f}, b={b:.2f}")
print(f"模型检验: 后验差比值C={C:.4f}, 小误差概率P={P:.4f}")

# 3. 读取未来人口数据
future_df = pd.read_excel('第一问预测结果.xlsx')
future_years = future_df['年份'].values
future_population = future_df['人口总数（万人）'].values

# 4. 预测未来生均经费
future_k = np.arange(len(expenditure), len(expenditure) + len(future_years))
future_expenditure = [predict_func(k) for k in future_k]

# 5. 计算增长率
growth_rates = [(future_expenditure[i] - future_expenditure[i - 1]) / future_expenditure[i - 1] * 100
                for i in range(1, len(future_expenditure))]
growth_rates.insert(0, np.nan)

# 6. 创建结果DataFrame
result_df = pd.DataFrame({
    '年份': future_years,
    '预测人口(万人)': future_population,
    '预测生均经费(元)': np.round(future_expenditure, 2),
    '增长率(%)': np.round(growth_rates, 2)
})
