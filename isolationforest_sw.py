import pandas as pd
from pycaret.anomaly import *
import plotly.graph_objects as go

#Anomaly detection in timeseries data with Isolation forest
headers = ['Time', 'lw_up', 'temp', 'Type']
df = pd.read_csv('SW_data.csv', header=0, names=headers, delimiter=',')

#Converting timestamp to datetime format
df['Time'] = pd.to_datetime(df['Time'])

df.drop(columns=['Type'], axis=1, inplace=True)
df.dropna(inplace=True)

#setting index as timestamp
df.set_index('Time', drop=True, inplace=True)

#aggregating time samples to hourly
df = df.resample('H').sum()

#creating features from timestamp
df['day'] = [i.day for i in df.index]
df['hour'] = [i.hour for i in df.index]

data = df

#setup of data
data_setup = setup(data, session_id=123)

#creating isolation forest model
iforest = create_model('iforest', fraction=0.05)


#assigning results of anomaly detection
iforest_results = assign_model(iforest)
print(iforest_results.head())
iforest_results.to_csv('unsupervised_anomaly_detection.csv')

#list of outlier dates as x values
outlier_dates = iforest_results[iforest_results['Anomaly'] == 1].index

#list of y_values as anomalies
y_values = [iforest_results.loc[i]['lw_up'] for i in outlier_dates]
y_values_temp = [iforest_results.loc[i]['temp'] for i in outlier_dates]


fig = go.Figure()
fig.add_trace(go.Scatter(x=iforest_results.index, y=data['lw_up'], mode='lines', name='leafwetness'))
fig.update_layout(legend_title_text='Smarter Weinberg', xaxis_title = "Date and Time",
                  yaxis_title="Leafwetness/Temperature" )
fig.add_scatter(x=iforest_results.index, y=data['temp'], name="temperature", legend="legend")

fig.add_trace(go.Scatter(x=outlier_dates, y= y_values, mode='markers', name='Anomaly_lw',
                         marker=dict(color='blue', size=10)))
fig.add_trace(go.Scatter(x=outlier_dates, y= y_values_temp, mode='markers', name='Anomaly_temp',
                         marker=dict(color='red', size=10)))

fig.show()

