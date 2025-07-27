import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

excel_file = pd.ExcelFile('国赛校赛第一问数据2.0.xlsx')
df = excel_file.parse('Sheet1')

df.set_index('年份', inplace=True)

def arima_forecast(data, order=(1, 1, 1), steps=22):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

# 预测高等教育毛入学率
higher_education_forecast = arima_forecast(df['高等教育毛入学率（%）'])

# 预测 GDP
gdp_forecast = arima_forecast(df['GDP（亿元）'])
