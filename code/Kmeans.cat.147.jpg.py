import cv2
import numpy as np
from sklearn.cluster import KMeans

class Format_data:
    def __init__(self, file_path_image : str) -> None:
        self.file_path_image = file_path_image

    def get_image(self) -> None:
        data = cv2.imread(self.file_path_image)
        self.data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return self.data    
    
    def format_dim(self) -> None:
        pixels = [[self.data[i,j] for j in range(self.data.shape[1])] for i in range(self.data.shape[0])]
        pixels = np.array(pixels).astype(np.float32)
        pixels /= 255.0
        self.data_new = pixels 

    def format_image_3D(self) -> np.ndarray:
        images = self.data_new
        pixels = images.reshape((images.shape[0] * images.shape[1],3))
        return pixels

class Kmeas:
    def __init__(self,n_cluters : int, image : np.ndarray) -> None:
        self.n_cluters = n_cluters
        self.image = image
    def run(self) -> None:
        self.model = KMeans(n_clusters=self.n_cluters, random_state=42).fit(self.image)
    def result_predict(self) -> None:
        labels = self.model.predict(self.image)
        return labels
    
#get size image
file_path_image = r"..\cat.147.jpg"
data = Format_data(file_path_image)

img = data.get_image()
data.format_dim()
pixels = data.format_image_3D()


#Run Kmeans
kmeans = Kmeas(2, pixels)
kmeans.run()
labels = kmeans.result_predict()

# tranform 3D array
image_new = np.zeros((img.shape[0],img.shape[1],1))

i = 0
j = 0
for pixel in labels:
    if (j >= img.shape[1]):
        j = 0
        i += 1
    image_new[i][j] = pixel
    j += 1

image0 = np.where(image_new == 0, 255, 0)
image1 = np.where(image_new == 1, 255, 0)

image0 = np.uint8(image0)
image1 = np.uint8(image1)

result_mask0 = cv2.bitwise_and(img,img,mask=image0)
result_mask1 = cv2.bitwise_and(img,img,mask=image1)

cv2.imshow("result mask ", result_mask1)
cv2.waitKey()

















