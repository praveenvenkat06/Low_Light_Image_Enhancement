import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import lib
 
# calculate frechet inception distance
def calculate_fid(value, Fid):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
  
	return value,Fid


def calculate():
	data = lib.data.get_training_set()
	num_samples = 50
	total1 = total2 = total3 = 0
	for i in range(num_samples):
		real_image, noisy_image, generated_image = data[i]

		fid1 = calculate_fid(real_image, real_image)
		fid2 = calculate_fid(generated_image, real_image)
		fid3 = calculate_fid(noisy_image, real_image)

		total1 += fid1
		total2 += fid2
		total3 += fid3

	avg1 = total1 / num_samples
	avg2 = total2 / num_samples
	avg3 = total3 / num_samples

	X = [1, 2, 3]
	Y = [avg1, avg2, avg3]

	plt.plot(X, Y)









































# define two collections of activations

act1 = random(10*2048)
act1 = act1.reshape((10,2048))
act2 = random(10*2048)
act2 = act2.reshape((10,2048))
values_x = [1,2,3]
values_y = [0,40,300]

# fid between act1 and act1
value,fid = calculate_fid(values_x, values_y)
plt.plot(value,fid,color='Blue')
plt.xlabel('Image')
plt.ylabel('Average Fretchet Distance')
plt.show()