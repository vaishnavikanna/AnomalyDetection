import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import openpyxl

#Data Generation

#df = pd.read_csv('Hive17.csv', usecols=['T17', 'RH17'])

contamination = 0.05  # percentage of outliers
n_train = 500  # number of training points
n_test = 500  # number of testing points
n_features = 6  # number of features
X_train, X_test, y_train, y_test = generate_data(
    n_train=n_train,
    n_test=n_test,
    n_features=n_features,
    contamination=contamination,
    random_state=123)


#Training the knn model
num_neighbors = 5
knn_model = NearestNeighbors(n_neighbors=num_neighbors)
knn_model.fit(X_train)

#Compute distances of test data to nearest neighbors in train data
distances, indices = knn_model.kneighbors(X_test)

#average distances to nearest neighbours of each test data point
average_dist = np.mean(distances, axis=1)

#threshold
threshold = np.percentile(average_dist, 95)

#Identifying anomalies with a threshold value
anomalies = X_test[average_dist > threshold]

#print(anomalies)
#plot
plt.plot(average_dist)
plt.xlabel('Test Data Point')
plt.ylabel('Distance to k-Nearest Neighbors')
plt.title('KNN Anomaly Detection')

# Mark the anomalies in the plot
for i in range(len(X_test)):
    if average_dist[i] > threshold:
        plt.plot(i, average_dist[i], 'ro')

# Plot the correct values
correct_values = X_test
plt.plot(correct_values, 'g')

plt.show()


