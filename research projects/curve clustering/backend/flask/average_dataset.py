import numpy as np
import os
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import cosine_similarity
import statistics
from scipy.stats import spearmanr
import csv
import time
from sklearn.preprocessing import StandardScaler

weights = [1,1,1,1]

def Standardization_preprocessing(training_output_list):
	scaler = StandardScaler()
	scaler.fit(training_output_list)
	return ((scaler.transform(training_output_list)).tolist(), scaler)

def weighted_euclidean_dis(a, b):
	global weights
	sum_list = 0
	if type(a) == list:
		list_length = len(a)
	else:
		list_length = a.shape[0]
	for i in range(0, list_length):
		sum_list = sum_list + ((a[i]-b[i])*(a[i]-b[i])*weights[i])
	return np.sqrt(sum_list)

def cluster_representative(training_input_list, training_set_labels, num_clusters):
	print("finding representatives...")
	similar_simulations = []
	representatives = []
	num_of_labels = num_clusters
	num_of_simulations = len(training_input_list)
	print(num_of_simulations)
	for label_value in range(0, num_of_labels):
		similar_simulations.append([])
		representatives.append([])

	for simulation_index in range(0, num_of_simulations):
		simulation_label = training_set_labels[simulation_index]
		simulation_input = training_input_list[simulation_index]
		similar_simulations[simulation_label].append(simulation_input)

	for label_v in range(0, num_of_labels):
		num_of_vectors = len(similar_simulations[label_v])
		similar_matrix = np.array(similar_simulations[label_v])
		representative_inputs = ((np.sum(similar_matrix, axis = 0))/num_of_vectors).tolist()
		representatives[label_v] = representative_inputs

	return(similar_simulations, representatives)

def unique(list1): 
	unique_list = [] 
	for x in list1: 
		if x not in unique_list: 
			unique_list.append(x)
	print(len(unique_list))

def cluster_plot(clusters_substrate, clusters_mpool, clusters_cpool, clusters_maxpoly, num_clusters):
	print("drawing plot...")
	for i in range(0, num_clusters):
		num_of_instances = len(clusters_substrate[i])
		min_sub = min(clusters_substrate[i])
		max_sub = max(clusters_substrate[i])
		min_m = min(clusters_mpool[i])
		max_m = max(clusters_mpool[i])
		min_c = min(clusters_cpool[i])
		max_c = max(clusters_cpool[i])
		min_my = min(clusters_maxpoly[i])
		max_my = max(clusters_maxpoly[i])
		sub_range = np.linspace(min_sub, max_sub, 40)
		m_range = np.linspace(min_m, max_m, 40)
		c_range = np.linspace(min_c, max_c, 40)
		my_range = np.linspace(min_my, max_my, 40)
		plt.hist(clusters_substrate[i], bins=sub_range, facecolor="cyan", alpha=0.3, label = 'substrate')
		plt.xlabel("value")
		plt.ylabel("num of instances")
		plt.title("distribution of substrate for cluster " + str(i) + "\nstd = " + str(statistics.stdev(clusters_substrate[i])) if num_of_instances > 1 else "None" )
		plt.legend(loc='upper right')
		plt.savefig('../pngs/substrate_cluster' + str(i) + '.png')
		plt.clf()

		plt.hist(clusters_mpool[i], bins=m_range, facecolor="violet", alpha=0.3, label = 'mpool')
		plt.xlabel("value")
		plt.ylabel("num of instances")
		plt.title("distribution of mpool for cluster " + str(i) + "\nstd = " + str(statistics.stdev(clusters_mpool[i])) if num_of_instances > 1 else "None" )
		plt.legend(loc='upper right')
		plt.savefig('../pngs/mpool_cluster' + str(i) + '.png')
		plt.clf()

		plt.hist(clusters_cpool[i], bins=c_range, facecolor="orange", alpha=0.3, label = 'cpool')
		plt.xlabel("value")
		plt.ylabel("num of instances")
		plt.title("distribution of cpool for cluster " + str(i) + "\nstd = " + str(statistics.stdev(clusters_cpool[i])) if num_of_instances > 1 else "None" )
		plt.legend(loc='upper right')
		plt.savefig('../pngs/cpool_cluster' + str(i) + '.png')
		plt.clf()

		plt.hist(clusters_maxpoly[i], bins=my_range, facecolor="palegreen", alpha=0.3, label = 'maxpoly')
		plt.xlabel("value")
		plt.ylabel("num of instances")
		plt.title("distribution of maxpoly for cluster " + str(i) + "\nstd = " + str(statistics.stdev(clusters_maxpoly[i])) if num_of_instances > 1 else "None" )
		plt.legend(loc='upper right')
		plt.savefig('../pngs/maxpoly_cluster' + str(i) + '.png')
		plt.clf()
		'''
		substrate_array = np.reshape(np.array(clusters_substrate[i]), (len(clusters_substrate[i]), 1))
		mpool_array = np.reshape(np.array(clusters_mpool[i]), (len(clusters_mpool[i]), 1))
		cpool_array = np.reshape(np.array(clusters_cpool[i]), (len(clusters_cpool[i]), 1))
		maxpoly_array = np.reshape(np.array(clusters_maxpoly[i]), (len(clusters_maxpoly[i]), 1))

		std_substrate = list(np.reshape(Standardization_preprocessing(substrate_array)[0],(len(clusters_substrate[i]),)))
		std_mpool = list(np.reshape(Standardization_preprocessing(mpool_array)[0],(len(clusters_mpool[i]),)))
		std_cpool = list(np.reshape(Standardization_preprocessing(cpool_array)[0],(len(clusters_cpool[i]),)))
		std_maxpoly = list(np.reshape(Standardization_preprocessing(maxpoly_array)[0],(len(clusters_maxpoly[i]),)))
		
		max_all = max([max(std_substrate), max(std_maxpoly), max(std_mpool), max(std_cpool)])
		min_all = min([min(std_substrate), min(std_mpool), min(std_maxpoly), min(std_cpool)])
		range_all = np.linspace(min_all, max_all, 40)
		'''
		max_all = max([max(clusters_substrate[i]), max(clusters_mpool[i]), max(clusters_cpool[i]), max(clusters_maxpoly[i])])
		min_all = min([min(clusters_substrate[i]), min(clusters_mpool[i]), min(clusters_cpool[i]), min(clusters_maxpoly[i])])
		range_all = np.linspace(min_all, max_all, 80)
		plt.hist(clusters_substrate[i], bins=range_all, facecolor="cyan", alpha=0.3, label = 'substrate')
		plt.hist(clusters_mpool[i], bins=range_all, facecolor="violet", alpha=0.3, label = 'mpool')
		plt.hist(clusters_cpool[i], bins=range_all, facecolor="orange", alpha=0.3, label = 'cpool')
		plt.hist(clusters_maxpoly[i], bins=range_all, facecolor="palegreen", alpha=0.3, label = 'maxpoly')
		plt.xlabel("value")
		plt.ylabel("num of instances")
		plt.title("distribution of all four parameters for cluster " + str(i))
		plt.legend(loc='upper right')
		plt.savefig('../pngs/all_cluster' + str(i) + '.png')
		plt.clf()




def clustering_with_knn(training_input_list, training_output_list, num_clusters, num_k, testing_output_list):
	similar_simulation_result = []
	suggested_inputs = []
	KNeighbors_suggested_inputs = [[] for _ in range(0, len(testing_output_list))]
	KNeighbors_suggested_inputs_distance = [[] for _ in range(0, len(testing_output_list))]
	clusters_substrate = [[] for _ in range(0, num_clusters)]
	clusters_mpool = [[] for _ in range(0, num_clusters)]
	clusters_cpool = [[] for _ in range(0, num_clusters)]
	clusters_maxpoly = [[] for _ in range(0, num_clusters)]

	print("training KNN and KMeans...")
	(stded_training_output, scaler) = Standardization_preprocessing(training_output_list)
	kmeans = KMeans(n_clusters = num_clusters)
	training_set_labels = kmeans.fit_predict(stded_training_output)

	for train_instance_index in range(0, len(training_set_labels)):
		clusters_substrate[training_set_labels[train_instance_index]].append(training_input_list[train_instance_index][0])
		clusters_mpool[training_set_labels[train_instance_index]].append(training_input_list[train_instance_index][1])
		clusters_cpool[training_set_labels[train_instance_index]].append(training_input_list[train_instance_index][2])
		clusters_maxpoly[training_set_labels[train_instance_index]].append(training_input_list[train_instance_index][3])
	cluster_plot(clusters_substrate, clusters_mpool, clusters_cpool, clusters_maxpoly, num_clusters)

	knn = KNeighborsClassifier(n_neighbors = num_k, weights = 'distance', metric=weighted_euclidean_dis)
	knn.fit(stded_training_output, training_set_labels)
	# indices of similar_simulations and representatives are actually labels
	(similar_simulations, representatives) = cluster_representative(training_input_list, training_set_labels, num_clusters)

	stded_testing_output = (scaler.transform(testing_output_list)).tolist()
	testing_set_labels = knn.predict(stded_testing_output)

	nearest_neighbor_indices_list = (knn.kneighbors(stded_testing_output, n_neighbors = 3, return_distance=False)).tolist()
	for testing_index in range(0, len(testing_output_list)):
		nearest_neighbor_index1 = nearest_neighbor_indices_list[testing_index][0]
		nearest_neighbor_index2 = nearest_neighbor_indices_list[testing_index][1]
		nearest_neighbor_index3 = nearest_neighbor_indices_list[testing_index][2]
		KNeighbors_suggested_inputs[testing_index].append(training_input_list[nearest_neighbor_index1])
		KNeighbors_suggested_inputs[testing_index].append(training_input_list[nearest_neighbor_index2])
		KNeighbors_suggested_inputs[testing_index].append(training_input_list[nearest_neighbor_index3])
		KNeighbors_suggested_inputs_distance[testing_index].append(float(weighted_euclidean_dis(stded_training_output[nearest_neighbor_index1], stded_testing_output[testing_index])))
		KNeighbors_suggested_inputs_distance[testing_index].append(float(weighted_euclidean_dis(stded_training_output[nearest_neighbor_index2], stded_testing_output[testing_index])))
		KNeighbors_suggested_inputs_distance[testing_index].append(float(weighted_euclidean_dis(stded_training_output[nearest_neighbor_index3], stded_testing_output[testing_index])))


	for values in testing_set_labels:
		similar_simulation_result.append(similar_simulations[values])
		suggested_inputs.append(representatives[values])

	return(similar_simulation_result, suggested_inputs, KNeighbors_suggested_inputs, KNeighbors_suggested_inputs_distance)



def read_data(header_list, input_list, output_list):
	print("reading data...")
	data_path = '../cms_viewer-latin/data.csv'
	rows = []
	with open(data_path, 'r') as csvfile: 
		csvreader = csv.reader(csvfile) 
		header_list = next(csvreader) 
		for row in csvreader: 
			input_list.append([float(i) for i in row[3:7]])
			output_list.append([float(j) for j in row[7:11]])


if __name__ == "__main__":
	
	input_list = []
	output_list = []
	header_list = []

	read_data(header_list, input_list, output_list)

	training_input_list = input_list[0:7000]
	training_output_list = output_list[0:7000]
	testing_output_list = output_list[7000:7500]
	actual_input_list = input_list[7000:7500]

	(similar_simulation_result, suggested_inputs, KNeighbors_suggested_inputs, KNeighbors_suggested_inputs_distance) = clustering_with_knn(training_input_list, training_output_list, 100, 3, testing_output_list)

	print("writing results...")

	with open("../results/similar_simulation_inputs.csv", "w", newline="") as simulations_file:
		writer = csv.writer(simulations_file)
		writer.writerows(similar_simulation_result)

	with open("../results/suggested_inputs.csv", "w", newline="") as suggested_inputs_file:
		writer = csv.writer(suggested_inputs_file)
		writer.writerows(suggested_inputs)

	with open("../results/KNeighbors_suggested_inputs.csv", "w", newline="") as KNeighbors_suggested_inputs_file:
		writer = csv.writer(KNeighbors_suggested_inputs_file)
		writer.writerows(KNeighbors_suggested_inputs)

	with open("../results/KNeighbors_suggested_inputs_distance.csv", "w", newline="") as KNeighbors_suggested_inputs_distance_file:
		writer = csv.writer(KNeighbors_suggested_inputs_distance_file)
		writer.writerows(KNeighbors_suggested_inputs_distance)

	with open("../results/actual_inputs.csv", "w", newline="") as actual_inputs_file:
		writer = csv.writer(actual_inputs_file)
		writer.writerows(actual_input_list)
	




