import numpy as np
import os
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import cosine_similarity
import statistics
from scipy.stats import spearmanr
import csv
import time
from sklearn.preprocessing import StandardScaler
import similaritymeasures
from sklearn.metrics.pairwise import pairwise_distances
import math
'''
weights = [1,1,0.000000001]

def weighted_euclidean_dis(a, b):
	global weights
	sum_list = 0
	list_length = a.shape[0]
	print(a.shape)
	print("ahhh")
	print(b.shape)
	for i in range(0, list_length):
		sum_list = sum_list + ((a[i]-b[i])*(a[i]-b[i])*weights[i])
	return np.sqrt(sum_list)

output = [[2,490,10000], [2,500,50000], [3,700,70000]]
output_labels = [1,2,3]
test = [[2,490,50000]]

knn = KNeighborsClassifier(n_neighbors = 1, metric=weighted_euclidean_dis)
knn.fit(output, output_labels)
print(knn.predict(test))
'''
'''
def Standardization_preprocessing(training_output_list):
	scaler = StandardScaler()
	scaler.fit(training_output_list)
	return ((scaler.transform(training_output_list)).tolist(), scaler)

training_output_list = [[1,100],[10,1],[5,20]]
a,b = Standardization_preprocessing(training_output_list)
print(a)
print(b.transform([[10,1]]))
'''
'''
a = [[],[]]

b = np.sum(a, axis = 0)
print((b/2).tolist())

a[0] = [1,2,4]
print(a)
'''
'''
from scipy import signal
sig = np.array([0,0,0,1,2,3,2,1])

sig_noise = np.array([0,1,4,1,0])
corr = signal.correlate( sig_noise, sig)
'''

'''
import matplotlib.pyplot as plt
clock = np.arange(64, len(sig), 128)
fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
'''
'''
ax_orig.plot(clock, sig[clock], 'ro')
'''
'''
x = np.random.random(100)
y = np.random.random(100)
exp_data = np.zeros((100, 2))


# Generate random numerical data
x = np.random.random(100)
y = np.random.random(100)
num_data = np.zeros((100, 2))

print(exp_data)
'''
'''
from scipy.spatial import distance
a = [[0,0],[1,3],[4,2]]
b = [[0,0],[0,2],[7,5]]
c = distance.cdist(a, b, 'cosine')
d = distance.cdist([[0.4,0.53]],[[0.22,0.38]],'euclidean')
print(c)
print(d)
from sklearn.metrics import pairwise_distances
data = np.array([0.40,0.53,0.22,0.38,0.35,0.32,0.26,0.19,0.08,0.41,0.45,0.30]).reshape(6,2)
initial_distances = pairwise_distances(data,metric='euclidean')
print(initial_distances)

e = np.arange(12).reshape(3, 4)
print(e)
f = np.delete(e, 1, 0)
g = np.delete(f, 1, 1)
print(g)
e = g
print(e)
'''
'''
a = [0.5, 1, 3, 4, 3, 2]
b = [4, 3, 3, 2, 1.1, 1]
centerx = sum(a)/len(a)
centery = sum(b)/len(b)
fig, (plt1, plt2) = plt.subplots(1, 2)
plt1.plot(a, b, label = 'path')
plt1.scatter(centerx, centery, color = 'red', label = 'center')
dis = []
x = []
import math
from scipy.interpolate import interp1d
for i in range(0, len(a)):
	plt1.plot([centerx, a[i]], [centery, b[i]], color = 'orange', label = 'distance'+str(i))
	dis.append(math.sqrt((a[i] - centerx)**2 + (b[i] - centery)**2))
	x.append(i)
x_smooth = np.linspace(min(x), max(x), num = 100)
f = interp1d(x, dis)
plt2.plot(x_smooth, f(x_smooth), color = 'red', label = 'signal')
plt2.legend()
plt1.legend()
fig.suptitle('global center', fontsize=30)
plt.show()
print(dis)
'''
'''
a = [0.5, 1, 3, 4, 3, 2]
b = [4, 3, 3, 2, 1.1, 1]
centerx = sum(a[:3])/len(a[:3])
centery = sum(b[:3])/len(b[:3])
fig, ((plt1, plt2), (plt3, plt4)) = plt.subplots(2, 2)
plt1.plot(a, b, label = 'path')
plt1.scatter(centerx, centery, color = 'red', label = 'center')
dis = []
x = []
import math

for i in range(0, len(a[:3])):
	plt1.plot([centerx, a[i]], [centery, b[i]], color = 'orange', label = 'distance'+str(i))
	dis.append(math.sqrt((a[i] - centerx)**2 + (b[i] - centery)**2))
	x.append(i)
plt2.set_xlim([0, 10]) 
plt2.set_ylim([0, 10]) 
plt2.plot(x, dis, color = 'red', label = 'signal')
plt2.legend()
plt1.legend()

centerx = sum(a[1:4])/len(a[1:4])
centery = sum(b[1:4])/len(b[1:4])
plt3.plot(a, b, label = 'path')
plt3.scatter(centerx, centery, color = 'red', label = 'center')
import math

for i in range(1, len(a[1:4])+1):
	plt3.plot([centerx, a[i]], [centery, b[i]], color = 'orange', label = 'distance'+str(i))
	dis.append(math.sqrt((a[i] - centerx)**2 + (b[i] - centery)**2))
	x.append(i+2)
plt4.set_xlim([0, 10]) 
plt4.set_ylim([0, 10]) 
plt4.plot(x, dis, color = 'red', label = 'signal')
plt3.legend()
plt4.legend()

fig.suptitle('local center', fontsize=30)
plt.show()
print(dis)
'''
'''
a = np.zeros([2,2])
a[1,1] = 3
print(a)
'''
'''
from scipy import signal
sig = np.array([1,2,3,2,1,1])

sig_noise = np.array([0,1,1,2,4,1])
sig_noise2 = np.array([1,4,2,1,1,0])
corr = list(signal.correlate(sig, sig_noise2))
print(corr)
print(corr.index(max(corr)))
def align_signals(sig1, sig2):
    corr = list(signal.correlate(sig1, sig2))
    max_index = corr.index(max(corr))
    middle_index = len(corr)//2
    if max_index == middle_index:
        return(sig1, sig2)
    elif max_index < middle_index:
        return(sig1[0:max_index+1],sig2[len(sig2)-(max_index+1):])
    else:
        mod_index = max_index - middle_index
        return(sig1[mod_index:], sig2[0:len(sig2)-mod_index])
print(align_signals(sig, sig_noise2))
'''



'''
print(list(set([0, 0, 0, 0, 4, 4, 6, 0, 0, 4, 0, 4, 12, 0, 0, 0, 0, 4, 0, 12])))
'''
'''
from scipy import signal
from statistics import mean
sig = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.2018504251546631, 0.6666666666666667, 1.6666666666666667, 1.2018504251546631, 1.3333333333333333, 1.2018504251546631, 1.6666666666666667, 0.6666666666666667, 1.2018504251546631, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

sig_noise = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.2018504251546631, 0.6666666666666667, 1.6666666666666667, 1.2018504251546631, 1.3333333333333333, 1.2018504251546631, 1.6666666666666667, 0.6666666666666667, 1.2018504251546631]
a = [i - mean(sig) for i in sig]
b = [i - mean(sig_noise) for i in sig_noise]
corr = list(signal.correlate([i-mean(sig) for i in sig], [j-mean(sig_noise) for j in sig_noise]))
max_index = corr.index(max(corr))
print(corr.index(max(corr)))
print(len(corr))
plt.plot(sig[0:max_index+1])
plt.plot(sig_noise[len(sig_noise)-(max_index+1):])
plt.show()
'''
'''
together = [(2,1),(1,2),(3,3)]
together.sort(key = lambda x: x[0])
print(together)
'''
'''
def sort_together(curve1, curve2):
	together = []
	for i in range(0, len(curve1)):
		together.append((curve1[i], curve2[i]))
	together.sort(key = lambda x: x[0])
	a = []
	b = []
	for i in range(0, len(together)):
		a.append(together[i][0])
		b.append(together[i][1])
	return [a,b] 
a = [3,2,6,1,7]
b = [62,54,1,56,3]
print(sort_together(a,b))
'''
'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math

x1 = 0
y1 = -math.sqrt(3)
x2 = -3
y2 = 0
x3 = 0
y3 = math.sqrt(3)
x4 = 3
y4 = 0


fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
c1 = plt.Circle((0, 0), math.sqrt(3), fill = False, alpha = 0.1)
c2 = plt.Circle((0, 0), 3, fill = False, alpha = 0.1)
center = plt.plot(0, 0,'.', color = 'black')
l1 = plt.Line2D((x1, x2), (y1, y2), lw=2.5)
l2 = plt.Line2D((x2,x3),(y2,y3),lw=2.5)
l3 = plt.Line2D((x3,x4),(y3,y4),lw=2.5)

def init():
	ax.add_patch(c1)
	ax.add_patch(c2)
	ax.add_artist(l1)
	ax.add_artist(l2)
	ax.add_artist(l3)
	return c1, c2, l1, l2, l3, 

def animate(i):
	global x1, y1, x2, y2, x3, y3, x4, y4

	x1 = math.sqrt(3) * np.sin(np.radians(180) + np.radians(3*i))
	y1 = math.sqrt(3) * np.cos(np.radians(180) + np.radians(3*i))
	x2 = 3 * np.sin(np.radians(-90) + np.radians(i))
	y2 = 3 * np.cos(np.radians(-90)+ np.radians(i))
	x3 = math.sqrt(3) * np.sin(np.radians(3*i))
	y3 = math.sqrt(3) * np.cos(np.radians(3*i))
	x4 = 3 * np.sin(np.radians(90) + np.radians(i))
	y4 = 3 * np.cos(np.radians(90) + np.radians(i))
	l1.set_data((x1, x2), (y1, y2))
	l2.set_data((x2, x3), (y2, y3))
	l3.set_data((x3, x4), (y3, y4))

	return c1, c2, l1, l2, l3, 

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=16, blit=True)


plt.show()
'''
'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math

x1 = 0
y1 = -math.sqrt(3)
x2 = -3
y2 = 0
x3 = 0
y3 = math.sqrt(3)
x4 = 3
y4 = 0


fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
c1 = plt.Circle((0, 0), math.sqrt(3), fill = False, alpha = 0.1)
c2 = plt.Circle((0, 0), 3, fill = False, alpha = 0.1)
center = plt.plot(0, 0,'.', color = 'black')
l1 = plt.Line2D((x1, x2), (y1, y2), lw=2.5)
l2 = plt.Line2D((x2,x3),(y2,y3),lw=2.5)
l3 = plt.Line2D((x3,x4),(y3,y4),lw=2.5)

def init():
	ax.add_patch(c1)
	ax.add_patch(c2)
	ax.add_artist(l1)
	ax.add_artist(l2)
	ax.add_artist(l3)
	return c1, c2, l1, l2, l3, 

def animate(i):
	global x1, y1, x2, y2, x3, y3, x4, y4

	x1 = math.sqrt(3) * np.sin(np.radians(180) + np.radians(i))
	y1 = math.sqrt(3) * np.cos(np.radians(180) + np.radians(i))
	x2 = 3 * np.sin(np.radians(-90) + np.radians(i))
	y2 = 3 * np.cos(np.radians(-90)+ np.radians(i))
	x3 = math.sqrt(3) * np.sin(np.radians(i))
	y3 = math.sqrt(3) * np.cos(np.radians(i))
	x4 = 3 * np.sin(np.radians(90) + np.radians(i))
	y4 = 3 * np.cos(np.radians(90) + np.radians(i))
	l1.set_data((x1, x2), (y1, y2))
	l2.set_data((x2, x3), (y2, y3))
	l3.set_data((x3, x4), (y3, y4))

	return c1, c2, l1, l2, l3, 

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=16, blit=True)


plt.show()
'''
'''
import math
a = [1,2,3,4]
b = [0,60/360, 120/360, 0]
plt.plot(a, b, color = 'm')
plt.show()
'''
'''
def find_minimum_distance(input, number_of_curves):
	current_min = float('inf')
	row = float('inf')
	col = float('inf')

	for i in range(0, number_of_curves):
		for j in range(0, i+1):
			if(input[i][j]<=current_min):
				current_min = input[i][j]
				row = i
				col = j
	return (row, col)

def find_clusters(input,method_name):
	all_result = []
	current_result = []
	number_of_curves = input.shape[0]
	np.fill_diagonal(input,float('inf'))

	for i in range(number_of_curves):
		current_result.append(i)
        
	all_result.append(current_result.copy())

	for _ in range(0, number_of_curves-1):
		row,col = find_minimum_distance(input, number_of_curves)
		if(method_name == "single"):
			for i in range(0,number_of_curves):
				if(i != col and i != row):
					merged_distance = min(input[col][i],input[row][i])
					input[col][i] = merged_distance
					input[i][col] = merged_distance

		elif(method_name=="complete"):
			for i in range(0,number_of_curves):
				if(i != col and i != row):
					merged_distance = max(input[col][i],input[row][i])
					input[col][i] = merged_distance
					input[i][col] = merged_distance

		elif(method_name == "average"):
			for i in range(0,number_of_curves):
				if(i != col and i != row):
					merged_distance = (input[col][i]+input[row][i])/2
					input[col][i] = merged_distance
					input[i][col] = merged_distance
        
		temp_mat1 = np.delete(input, row, 0)
		temp_mat2 = np.delete(temp_mat1, row, 1)
		input = temp_mat2
		number_of_curves = input.shape[0]
       
		for n in range(len(current_result)):
			if(current_result[n] == row):
				current_result[n] = col
			elif(current_result[n] > row):
				current_result[n] -= 1
		all_result.append(current_result.copy())

	return all_result

data = np.array([0.40,0.53,0.22,0.38,0.35,0.32,0.26,0.19,0.08,0.41,0.45,0.30]).reshape(6,2)
initial_distances = pairwise_distances(data,metric='euclidean')
clusters = find_clusters(initial_distances,'single')
print(clusters)
'''
'''
def curve_to_signal(curve, signal_type):
    print(curve)
    curve_length = len(curve[0])
    if signal_type == 'global_center':
        sig = []
        centerx = sum(curve[0])/curve_length
        centery = sum(curve[1])/curve_length
        for i in range(0, curve_length):
            sig.append(math.sqrt((curve[0][i] - centerx)**2 + (curve[1][i] - centery)**2))
        return sig
    elif signal_type == 'local_center':
        sig = []
        ratio = 5
        local_length = curve_length//ratio
        startp = 0
        endp = startp + local_length
        while endp <= curve_length:
            centerx = sum(curve[0][startp:endp])/ratio
            centery = sum(curve[1][startp:endp])/ratio
            for i in range(startp, endp):
                sig.append(math.sqrt((curve[0][i] - centerx)**2 + (curve[1][i] - centery)**2))
            startp += 1
            endp += 1
        return sig
    elif signal_type == 'velocity_center':
        sig = []
        ratio_v = 5
        local_length = curve_length//ratio_v
        startp = 0
        endp = startp + local_length
        while endp <= curve_length:
            centerx = curve[0][startp]
            centery = curve[1][startp]
            for i in range(startp, endp):
                sig.append(math.sqrt((curve[0][i] - centerx)**2 + (curve[1][i] - centery)**2))
            startp += 1
            endp += 1
        return sig
    else:
        raise ValueError('signal type error')
'''
