from sklearn.cluster import KMeans

import cv2

img = cv2.imread("./temp/Black_Footed_Albatross_0009_34.jpg")


pixels = img.reshape(-1,3)
print(pixels.shape)
kmeans = KMeans(n_clusters= 2,random_state=0)

kmeans.fit(pixels)
labels = kmeans.cluster_centers_[kmeans.labels_] 


# print(labels)
new_img = labels.reshape(img.shape)
cv2.imwrite("output.jpg",new_img)


