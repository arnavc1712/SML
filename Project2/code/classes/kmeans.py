import numpy as np
import scipy.io
import random
import sys
import os
from collections import defaultdict
from matplotlib import pyplot as plt
from copy import deepcopy

cur_dir = os.path.dirname(os.path.abspath(__file__))


## Class for KMeans
## Parameters: Number of clusters, data path, Strategy to initialize clusters

class KMeans():
    def __init__(self,num_clusters,data_path,random_seed,max_iterations=200,strategy=1):
        random.seed(random_seed)
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.strategy = strategy
        self.center_points_dict = defaultdict(int)
        self.all_samples_data = self.load_data(data_path)
        self.all_centers = []
        if strategy==1: ## Using Random Sample to ensure k number of indexes are unique
            idxes = random.sample(set(range(len(self.all_samples_data))),self.num_clusters) 
            self.all_centers = self.all_samples_data[idxes]
        elif strategy==2: 
            rand_idx = random.randint(0,len(self.all_samples_data)-1) ## Choosing first point index according to strategy 2
            self.all_centers.append(self.all_samples_data[rand_idx])
            chosen_idx = [rand_idx]
#             print(rand_idx)
            
            for k in range(1,self.num_clusters): ## Choosing remaining points according to strategy 2
                distances = self.calc_point_distances()
                for dist,idx in distances: ## Looping through the distances sorted in descending order and choosing 
                                           ## centroid only if distance is maximum from remaining centroids and
                                           ## centroid not already present in the all_centroid list
                    if idx not in chosen_idx:
#                         print(idx)
                        new_point = self.all_samples_data[idx]
                        chosen_idx.append(idx)
                        self.all_centers.append(new_point)
                        break
    
    
    def calc_point_distances(self):
        ## Returns average distance of every point to all the present centroids
        distances = []
        
        for i,points in enumerate(self.all_samples_data):
            curr_dist = []
            curr_dist = list(map(lambda x:self.euclidean_distance(points,x),self.all_centers))
            distances.append((np.average(np.array(curr_dist)),i))
        
#         print(np.argmax(distances))
        return sorted(distances,key=lambda x:x[0],reverse=True)
        
        
    def load_data(self,data_path):
        ## Loads data from the data path
        numpyfile= scipy.io.loadmat(data_path) 
        return numpyfile["AllSamples"]
    
    def objective_function(self):
        ## Calculates total objective function
        final_sum = 0
        for i in range(self.num_clusters):
            center_i = self.all_centers[i]
            data_points_i = self.center_points_dict[i]

            squared = np.square(center_i-data_points_i)
            sum_c = np.sum(squared)
            final_sum+=sum_c
        return final_sum
    
    def euclidean_distance(self,a,b):
        ## Calculates Euclidean distance
        return np.linalg.norm(a-b)
    
    def plot_points(self):
        group = []
        x = []
        y = []
        for k,i in self.center_points_dict.items():
            group.extend([k+1]*len(i))
            x.extend([elem[0] for elem in i])
            y.extend([elem[1] for elem in i])
        group,x,y = np.array(group),np.array(x),np.array(y)
        fig, ax = plt.subplots(figsize=(10,10))
        cdict = {k+1:k+1 for k in range(k)}
        for g in np.unique(group):
            ix = np.where(group == g)
            ax.scatter(x[ix], y[ix], label = g, s = 100,cmap="viridis")
        ax.legend()
        ax.set_title('K = {} Strategy = {}'.format(str(self.num_clusters),str(self.strategy)))
        # plt.savefig(os.path.join(cur_dir,"../images/scatter_plot_{}_{}".format(self.num_clusters,self.strategy)))
        plt.clf()
        # plt.show()
    
    def plot_centroids(self):
        group = []
        x = []
        y = []
        for k in range(self.num_clusters):
            group.append(k)
            x.append(self.all_centers[k][0])
            y.append(self.all_centers[k][1])
        group,x,y = np.array(group),np.array(x),np.array(y)
        fig, ax = plt.subplots(figsize=(10,10))
        cdict = {k+1:k+1 for k in range(k)}
#         min_x = min([x[0] for x in self.all_samples_data])
#         max_x = max([x[0] for x in self.all_samples_data])
#         min_y = min([x[1] for x in self.all_samples_data])
#         max_y = max([x[1] for x in self.all_samples_data])
#         plt.xlim([min_x-2, max_x+2])
#         plt.ylim([min_y-2, max_y+2])
        for g in np.unique(group):
            ix = np.where(group == g)
            ax.scatter(x[ix], y[ix], label = g, s = 100,cmap="viridis")
        ax.legend()
        plt.show()
    
    
    def kmeans(self):

        ## Main algorithm for KMeans. Stops either if the centroids converge first or if the maximum number of iterations are reached. 
        ## Whichever condition reaches first
        for i in range(self.max_iterations):
            new_centers = deepcopy(self.all_centers)
            self.center_points_dict = defaultdict(list)

            for points in self.all_samples_data:
                min_dist = sys.maxsize
                min_idx = None
                for cluster_idx,cluster in enumerate(self.all_centers):
                    dist = self.euclidean_distance(points,cluster)
                    if min_idx==None:
                        min_dist = dist
                        min_idx = cluster_idx
                        continue

                    if dist<min_dist:
                        min_dist = dist
                        min_idx = cluster_idx

                self.center_points_dict[min_idx].append(points)

            for cluster_idx,points in self.center_points_dict.items():
                new_centers[cluster_idx] = np.mean(points,axis=0)
            
            if np.array_equal(new_centers,self.all_centers):
                break
            else:
                self.all_centers = deepcopy(new_centers)

