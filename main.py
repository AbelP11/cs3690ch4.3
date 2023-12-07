import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

df = pd.read_csv("C:\Users\apara\Downloads\Credit Card Customer Data.csv")

X, y = load_iris(return_X_y=True)

sse = []
for k in range(1,11):
    km = KMeans(n_clusters=k, random_state=2)
    km.fit(X)
    sse.append(km.inertia_)

sns.set_style("whitegrid")
g=sns.lineplot(x=range(1,11), y=sse)

g.set(xlabel = "Number of cluster (k)",
      ylabel = "Sum Squared Error",
      title = "Elbow Method")



kmeans = KMeans(n_clusters = 3, random_state = 2)
kmeans.fit(X)

kmeans.cluster_centers_


pred = kmeans.fit_predict(X)
pred

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c = pred, cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center = center[:2]
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.xlabel("Avg_Credit_Limit")
plt.ylabel("Total_Credit_Cards")
plt.show()