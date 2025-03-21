
# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
## Date:

### AIM:

To implement SARIMA model using python.


### ALGORITHM:


1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model



### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
data = pd.read_csv('Temperature.csv')
data['date'] = pd.to_datetime(data['date'])
plt.plot(data['date'], data['temp'])
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Time Series')
plt.show()
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
check_stationarity(data['temp'])
plot_acf(data['temp'])
plt.show()
plot_pacf(data['temp'])
plt.show()
sarima_model = SARIMAX(data['temp'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
train_size = int(len(data) * 0.8)
train, test = data['temp'][:train_size], data['temp'][train_size:]
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()
prediction = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```

### OUTPUT:
![10 1](https://github.com/shaikSameerbasha5404/TSA_EXP10/assets/118707756/681feffd-149c-4841-aba8-7c25fb1edc06)


![10 2](https://github.com/shaikSameerbasha5404/TSA_EXP10/assets/118707756/0a74ec6b-3682-48e4-aa3c-eb6a6143c1e3)


![10 3](https://github.com/shaikSameerbasha5404/TSA_EXP10/assets/118707756/4cf8fba0-4c1b-458b-abf3-499794ec31b9)


![10 4](https://github.com/shaikSameerbasha5404/TSA_EXP10/assets/118707756/597d7c01-f0a8-4877-85dc-8797db59aaa9)


### RESULT:

Thus the program run successfully based on the SARIMA model.
