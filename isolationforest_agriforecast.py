import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import shap

#data preproceesing and feature engineering
headers = ['GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'DAY', 'TEMPERATURE_AVG', 'WINDSPEED',
           'VAPOURPRESSURE', 'PRECIPITATION', 'ET0', 'RADIATION']
df = pd.read_csv('WeatherData.csv', header=0, names=headers, delimiter=';', parse_dates=['DAY'])
df.drop(columns = ['GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE'], axis=1, inplace=True)
df['DAY'] = pd.to_datetime(df['DAY'])
df = df.dropna(how = 'any', axis = 0) #remove empty rows in any
df.set_index('DAY', drop=True, inplace=True)
data = df

#building the isolation forest model
model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.05), max_features=1.0)
model.fit(data.values)

results = data.copy()

#Predicting anomalies and scores
results['anomaly']= model.predict(data.values)
results['scores']=model.decision_function(data.values)
results.to_csv('unsupervised_anomaly_detection_agriforecast.csv')
print(results.head(20))

#plotting the anomalies
fig = px.scatter(results.reset_index(), x='DAY', y='TEMPERATURE_AVG', color='anomaly', title='AgriforecastEU')


fig.show()

#shap to understand the importance of features
#explainer = shap.Explainer(model, results)
#shap_values = explainer(results)
#shap.plots.beeswarm(shap_values)
#plotting the anomalies
