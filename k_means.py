# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:26:24 2020

@author: Achyuth Maddala Sitaram
"""
# Importing all the required libraries
import numpy as np
import scipy.io as sio
import random as rd
import matplotlib.pyplot as plt
from scipy.spatial import distance

#Loading data from the data set 
data = sio.loadmat('AllSamples.mat')  
cord = data['AllSamples']

#print (cord.shape)
#print (type(cord))
N = cord.shape[0]

#Function for Strategy2:Randomly initializing centroids
def random_centroids(cords,clusters):      
    rd.seed()
    centroids = np.zeros((clusters, 2))
    for i, j in enumerate(centroids):
        centroids[i] = cords[rd.randint(0,N-1)]    
    #print(centroids)
    #print(centroids.shape)
    return centroids

# clusters = 5
# centroids = random_centroids(cord,clusters)
# print(centroids.shape)
# print(centroids)

#Function to compute distance from each centroid to every coordinate from the dataset
def cent_dist(cords,centroids):    
    euclid_dist = np.array([]).reshape(N,0)
    for k in range(centroids.shape[0]):
        #print(centroids.shape[0])
        temp_dist=np.linalg.norm((cords-centroids[k]),axis=1)
        euclid_dist=np.c_[euclid_dist,temp_dist]
    return(euclid_dist)

# dist = cent_dist(cord,centroids)
# print(dist.shape)

#Function to assign 2D coordinates to their nearest centroid
def assign(dist):             
    cluster_array = np.reshape(np.zeros(N), (-1, 1))
    for index,i in enumerate(dist):
        position = np.argmin(dist[index])
        cluster_array[index] = position+1
    return(cluster_array)

# setting = assing(dist)
# print(setting.shape)

#Function to sub-list the entire coordinates into each sub-list based on the centroids the coordinates are assigned in assign fuction 
def sorted_cluster(cords,clusters,assignments):
    sorted_cluster = [[]for i in range(clusters)]   
    #a = npsorted_cluster#a = np.asarray(cord)
    for i in range(assignments.shape[0]):
        #print(setting[i][0])
        sorted_cluster[int(assignments[i])-1].append(cords[i])
    sorted_cluster= np.asarray(sorted_cluster)
    return (sorted_cluster)

# init_clusters = sorted_cluster(cord,clusters)
# print(init_clusters.shape)

#Function to calculate the means and update the centroids
def means(init_clusters,clusters,prev_centds):
    mean = []
    for idx,cluster in enumerate(init_clusters):
        if cluster:
            mean.append(np.mean(cluster, axis = 0))
        else: 
            mean.append(prev_centds[idx])
    mean = np.asarray(mean)
    return mean

# mean_prev = means(init_clusters,clusters)
# print(mean_prev)

#Function for Strategy 2: Farthest from the first
def farthest_first(cords,clusters):
    #print(c)
    centroids = []
    centroids.append(cords[rd.randint(0,N-1)])
    for j in range(1,clusters):
        temp_list = []
    #for each point in X, find the distance from all the previous place
        for i in range(len(cords)):
            dist = 0
            avg = 0
            for ind in range(len(centroids)):
                #dist += np.linalg.norm((X-c_ar[k]),axis=1)
                dist += distance.euclidean(cords[i],centroids[ind])
                avg += 1
            temp_list.append(dist/avg)
        pos = max(temp_list)
        idx = temp_list.index(pos)
        centroids.append(cords[idx])
        #X = np.delete(X,m,ax)
    #print(c_ar)
    return centroids

# #points obtained from farest function
# for c in range (2,11):
#     print(c)
#     zz = furthest_from_first(cord,c)
#     zz = np.asarray(zz)
#     print(zz)

#Driver code for computing graph of objective function vs # of clusters twice for Strategy 1: Random picking centroids
for a in range(2):
    #objective function
    obj = []
    clu = []
    for i in range(2,11):
        clu.append(i)
        #i = 3
        #print("********************")
        #print(i)
        #print('####################')
        centroids = random_centroids(cord,i)  
        mean_pres = np.copy(centroids)
        count = 0
        while True:
            mean_prev = np.copy(mean_pres)
            dist = cent_dist(cord,mean_prev)
            assigned = assign(dist)
            init_clusters = sorted_cluster(cord,i,assigned)
            mean_pres = means(init_clusters,i,mean_prev)
            #print(mean_pres)
            count = count + 1
            if np.array_equal(mean_pres,mean_prev) == True:
                break
        final_cents = np.copy(mean_pres)
        clusts = np.copy(init_clusters)
        final_cents = final_cents.tolist()
        #print(final_cents)
        dist = []
        for p in range(i):
            temp = 0
            ad = (np.asarray(clusts[p]))
            temp = np.sum(np.linalg.norm((ad-np.asarray(final_cents[p])),axis=1))
            #dist[0][b] += distance.euclidean(clusts[],final_cents[b])
            dist.append(temp)
        #print(dist)
        temp_obj = sum(dist)
        obj.append(temp_obj)
    print(clu)
    print(obj)
    plt.plot(clu,obj,label="objectve_function vs number of clusters")  
    plt.xlabel('Cluster points')
    plt.ylabel('Objective Function value')
    plt.title("Objectve function vs number of cluster points")
    plt.legend()
    plt.grid()
    plt.show()
    
#Driver code for computing graph of objective function vs # of clusters twice for Strategy 2: Farthest from the first
for a in range(2):
    #driver code farest
    obj = []
    clu = []
    for c in range(2,11):
        clu.append(c)
        #c = 3
        #print("**************************")
        #print(c)
        #print('##########################')
        far_pts = farthest_first(cord,c)
        far_pts = np.asarray(far_pts)
        #print(far_pts)
        mean_pres = np.copy(far_pts)
        count = 0
        while True:
            mean_prev = np.copy(mean_pres)
            dist = cent_dist(cord,mean_prev)
            assigned = assign(dist)
            init_clusters = sorted_cluster(cord,c,assigned)
            mean_pres = means(init_clusters,c,mean_prev)
            count = count +1
            if np.array_equal(mean_pres,mean_prev) == True:
                break
        #print(count)
        #print(mean_pres)
        final_cents = np.copy(mean_pres)
        clusts = np.copy(init_clusters)
        final_cents = final_cents.tolist()
        #print(final_cents)
        dist = []
        for p in range(c):
            temp = 0
            ad = (np.asarray(clusts[p]))
            temp = np.sum(np.linalg.norm((ad-np.asarray(final_cents[p])),axis=1))
            #dist[0][b] += distance.euclidean(clusts[],final_cents[b])
            dist.append(temp)
        #print(dist)
        temp_obj = sum(dist)
        obj.append(temp_obj)
    print(clu)
    print(obj)
    plt.plot(clu,obj,label="objectve_function vs number of clusters")  
    plt.xlabel('Cluster points')
    plt.ylabel('Objective Function value')
    plt.title("Objectve function vs number of cluster points")
    plt.legend()
    plt.grid()
    plt.show()
