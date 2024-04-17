# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply k means clustering for customer segmentation

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:Vasanthamukilan M
RegisterNumber:212222230167 
*/
```

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Mall_Customers_EX8.csv")
data

X=data[['Annual Income (k$)','Spending Score (1-100)']]
X

plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

k=5
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

colors=['r', 'g', 'b', 'c', 'm']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],
              color=colors[i], label=f'Cluster {i+1}')
  distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids [:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```
## Output:
## DATASET:
![Screenshot 2024-04-17 152746](https://github.com/Vasanthamukilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119559694/c9b386ce-9101-430b-b76c-a2c58dff66d2)

## GRAPH:
![Screenshot 2024-04-17 152858](https://github.com/Vasanthamukilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119559694/c6b9d599-8875-4d91-8a3a-c1f52a61e00b)

## K-Means:
![Screenshot 2024-04-17 152925](https://github.com/Vasanthamukilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119559694/d4a5fcbc-f185-4bbc-b914-db9784f867b0)



## CENTROID VALUE:
![Screenshot 2024-04-17 153211](https://github.com/Vasanthamukilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119559694/96f03374-e3af-46e7-ae6b-2b696fc3095b)

## K-Means CLUSTERING:
![Screenshot 2024-04-17 153224](https://github.com/Vasanthamukilan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119559694/fb495587-32df-43f7-87da-a047c2ad0261)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
