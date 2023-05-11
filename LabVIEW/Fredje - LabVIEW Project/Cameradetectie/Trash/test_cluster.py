from sklearn.cluster import KMeans
import numpy as np

coordinates = [
    (1, 2), (2, 3), (1, 3),
    (10, 11), (11, 12), (10, 12),
    (20, 21), (21, 22), (20, 22)
]

# Convert the list of tuples to a NumPy array
data = np.array(coordinates)

# Apply k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)

# Find the highest y value in each cluster
highest_y_coordinates = []
for cluster in range(3):
    cluster_points = data[kmeans.labels_ == cluster]
    highest_y_point = cluster_points[np.argmax(cluster_points[:, 1])]
    highest_y_coordinates.append(tuple(highest_y_point))

print(highest_y_coordinates)