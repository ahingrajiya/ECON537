import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.ar_model as smt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

df_all = pd.read_csv('DGS10.csv')
df = df_all.dropna()
df["Date"] = pd.to_datetime(df["observation_date"], format='%Y-%m-%d')
df.set_index('Date', inplace=True) 


ar1_model = smt.AutoReg(df['DGS10'].to_numpy(), lags=1)
ar1_fit = ar1_model.fit()
print("AR(1) Model Result Summary")
print(ar1_fit.summary())

ar2_model = smt.AutoReg(df['DGS10'].to_numpy(), lags=2)
ar2_fit = ar2_model.fit() 
print("AR(2) Model Result Summary")
print(ar2_fit.summary()) 

residuals_ar1 = df["DGS10"].to_numpy()[1:] - ar1_fit.fittedvalues
residuals_ar2 = df["DGS10"].to_numpy()[2:] - ar2_fit.fittedvalues

qtest_ar1 = sm.stats.acorr_ljungbox(residuals_ar1, lags=[10], return_df=True)
print("AR(1) Model Q-Test")
print(qtest_ar1)
qtest_ar2 = sm.stats.acorr_ljungbox(residuals_ar2, lags=[10], return_df=True)
print("AR(2) Model Q-Test")
print(qtest_ar2)

data_1 = df[(df['observation_date'] >= '1962-01-01') & (df['observation_date'] <= '1980-12-31')]
data_2 = df[(df['observation_date'] >= '1981-01-01') & (df['observation_date'] <= '2000-12-31')]

full_model = smt.AutoReg(df['DGS10'], lags=1)
full_model_result = full_model.fit()
full_model_residuals = full_model_result.resid
full_model_rss = np.sum(full_model_residuals ** 2)

data_1_model = smt.AutoReg(data_1['DGS10'], lags=1)
data_1_model_result = data_1_model.fit()
data_1_model_residuals = data_1_model_result.resid
data_1_model_rss = np.sum(data_1_model_residuals ** 2)

data_2_model = smt.AutoReg(data_2['DGS10'], lags=1)
data_2_model_result = data_2_model.fit()
data_2_model_residuals = data_2_model_result.resid
data_2_model_rss = np.sum(data_2_model_residuals ** 2)

print("Full Model RSS: ", full_model_rss)
print("Data 1 Model RSS: ", data_1_model_rss)
print("Data 2 Model RSS: ", data_2_model_rss)

f_stat = ((full_model_rss - (data_1_model_rss + data_2_model_rss)) / 2) / ((data_1_model_rss + data_2_model_rss) / (len(df) - 4))
print("F-Statistic for AR(1): ", f_stat)

f_crit = stats.f.ppf(0.95, 2, len(df) - 4)
print("Critical F-Value for AR(1): ", f_crit)

p_value = 1 - stats.f.cdf(f_stat, 2, len(df) - 4)
print("P-Value for AR(1): ", p_value)


full_model = smt.AutoReg(df['DGS10'], lags=2)
full_model_result = full_model.fit()
full_model_residuals = full_model_result.resid
full_model_rss = np.sum(full_model_residuals ** 2)

data_1_model = smt.AutoReg(data_1['DGS10'], lags=2)
data_1_model_result = data_1_model.fit()
data_1_model_residuals = data_1_model_result.resid
data_1_model_rss = np.sum(data_1_model_residuals ** 2)

data_2_model = smt.AutoReg(data_2['DGS10'], lags=2)
data_2_model_result = data_2_model.fit()
data_2_model_residuals = data_2_model_result.resid
data_2_model_rss = np.sum(data_2_model_residuals ** 2)

print("Full Model RSS: ", full_model_rss)
print("Data 1 Model RSS: ", data_1_model_rss)
print("Data 2 Model RSS: ", data_2_model_rss)

f_stat = ((full_model_rss - (data_1_model_rss + data_2_model_rss)) / 3) / ((data_1_model_rss + data_2_model_rss) / (len(df) - 6))
print("F-Statistic for AR(2): ", f_stat)

f_crit = stats.f.ppf(0.95, 3, len(df) - 6)
print("Critical F-Value for AR(2): ", f_crit)

p_value = 1 - stats.f.cdf(f_stat, 3, len(df) - 6)
print("P-Value for AR(2): ", p_value)

train_size = int(len(df) * 0.4)  
train, test = df['DGS10'][:train_size], df['DGS10'][train_size:]

err_ar1 = []
err_ar2 = []

for i in range(len(test)):
    train_data = pd.concat([train, test[:i]])
    ar1fit = smt.AutoReg(train_data, lags=1).fit()
    forecast_ar1 = ar1fit.predict(start=len(train_data), end=len(train_data))
    ar2fit = smt.AutoReg(train_data, lags=2).fit()
    forecast_ar2 = ar2fit.predict(start=len(train_data), end=len(train_data))
    actual_values= test.iloc[i]
    err_ar1.append(actual_values - forecast_ar1)
    err_ar2.append(actual_values - forecast_ar2)

errors_ar1 = np.array(err_ar1)
errors_ar2 = np.array(err_ar2)
rmse_ar1 = np.sqrt(sklm.mean_squared_error(np.zeros_like(err_ar1), err_ar1))
rmse_ar2 = np.sqrt(sklm.mean_squared_error(np.zeros_like(err_ar2), err_ar2))

print(f"RMSE for AR(1) model: {rmse_ar1}")
print(f"RMSE for AR(2) model: {rmse_ar2}")
