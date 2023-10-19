import pandas as pd
from pycaret.anomaly import *
import shap
from pycaret.classification import *

headers = ['GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'DAY', 'TEMPERATURE_AVG', 'WINDSPEED',
           'VAPOURPRESSURE', 'PRECIPITATION', 'ET0', 'RADIATION']
df = pd.read_csv('WeatherData.csv', header=0, names=headers, delimiter=';')
df.drop(columns = ['GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'DAY'], axis=1, inplace=True)
df['Anomaly_Target'] = [1 if x>=10.0 else 0 for x in df['PRECIPITATION']]
#print(df)
#pycaret

dataset = df

data = dataset.sample(frac=0.65, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
#print(dataset)

#setup of data
data_setup = setup(data = data, normalize=True, session_id=123, target = 'Anomaly_Target')

#creating logistic regression model
lr_model = create_model('lr')
print(lr_model)


lr_model_results = predict_model(lr_model)

#predicting anomaly score and label based on trained lr_model
lr_unseen = predict_model(lr_model, data=data_unseen)
print(lr_unseen.head())
lr_unseen.to_csv('supervised_anomaly_detection_df.csv')
#
# #explainer = shap.Explainer(iforest, data)
# #shap_values = explainer(data)
# #shap.plots.waterfall(shap_values[0])
#plotting the anomalies
#plot_model(lr_unseen, plot = 'auc')
