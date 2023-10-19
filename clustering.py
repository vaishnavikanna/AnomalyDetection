import pandas as pd
from pycaret.anomaly import *
import pycaret.clustering as pc
import shap


headers = ['GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE', 'DAY', 'TEMPERATURE_AVG', 'WINDSPEED',
           'VAPOURPRESSURE', 'PRECIPITATION', 'ET0', 'RADIATION']
df = pd.read_csv('WeatherData.csv', header=0, names=headers, delimiter=';')
df.drop(columns = ['GRID_NO', 'LATITUDE', 'LONGITUDE', 'ALTITUDE'], axis=1, inplace=True)
#print(df)
#pycaret

dataset = df

data = dataset.sample(frac=0.65, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

#setup of data
data_setup = pc.setup(data, normalize=True, session_id=123, ignore_features=['DAY'])

#creating kmeans model
clustering = pc.create_model('kmeans', num_clusters=7)
print(clustering)

#assigning results of anomaly detection
clustering_results = pc.assign_model(clustering)
print(clustering_results.head())
clustering_results.to_csv('unsupervised_anomaly_detection.csv')

#explainer = shap.Explainer(iforest, data)
#shap_values = explainer(data)
#shap.plots.waterfall(shap_values[0])
#plotting the anomalies
pc.plot_model(clustering, plot='cluster')
