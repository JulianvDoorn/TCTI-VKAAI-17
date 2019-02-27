#!/bin/env python3

import numpy as np
import random
import dataset
from vector import Vector
from typing import List, Optional, TypeVar, AnyStr, T, Tuple, Dict
from multiprocessing.pool import ThreadPool, ApplyResult
import matplotlib.pyplot as plt
import copy

## Typedef for Label, bound with any string-like object
#
# @brief Used by DataPoint to label points
Label = TypeVar("Label", bound=AnyStr)

class Cluster:
    last_cluster_id: int = 0

    @staticmethod
    def generate_random_vector(dimensions, limits):
        return Vector(*[random.uniform(*(limits[i])) for i in range(dimensions)])

    def __init__(self, dimensions, limits):
        self.id = Cluster.last_cluster_id + 1
        Cluster.last_cluster_id += 1
        self.centroid = Cluster.generate_random_vector(dimensions, limits)
        self.accumulator: Vector = Vector(*([0] * dimensions))
        self.point_count = 0
        self.centroid_history = [self.centroid]

    def update_centroid(self):
        if self.point_count != 0:
            self.centroid = Vector(*(a / self.point_count for a in self.accumulator))
        self.accumulator = Vector(*([0] * len(self.accumulator.values)))
        self.point_count = 0
        self.centroid_history.append(self.centroid)
        return (self.centroid_history[-2] - self.centroid).norm()

    def __repr__(self):
        return "Cluster " + str(self.id)

## Generic DataPoint class 
#
# @brief DataPoint is suitable for different machine learning applications. It
#        has the ability to run kNearestNeighbors algorithms using reference
#        input data.
class DataPoint:
    ## Converts a List[T] of any applicable T to a List[DataPoint]
    #
    # @brief Takes a List[T] dataset and an optional List[Label]. For each index
    #        i, a datapoint from dataset and a label from labels are bound to
    #        eachother.
    #
    # @param List[T] dataset to read from
    # @param Optional[List[Label]] labels to bind with dataset at index i
    @staticmethod
    def from_dataset(dataset: List[T], labels: Optional[List[Label]]=None) -> List['DataPoint']:
        datapoints: List[DataPoint] = []

        if labels is None:
            labels = [None] * len(dataset)

        for i in range(len(dataset)):
            temp = [ ]

            for col in dataset[i]:
                temp.append(col)

            datapoints.append(DataPoint(labels[i], Vector(*temp)))

        return datapoints

    ## Returns the most dominant label from List[Label] labels
    #
    # @param List[Label] labels to take the most dominant from
    @staticmethod
    def get_dominant_label(labels: List[Label]) -> Label:
        counted_labels = { }

        for l in labels:
            if l not in counted_labels.keys():
                counted_labels[l] = 1
            else:
                counted_labels[l] += 1

        return sorted(counted_labels, key=(lambda key: counted_labels[key]), reverse=True)[0]
        
    @staticmethod
    def calculate_limits( data: List['DataPoint'], dimensions: int) -> List[Tuple[int, int]]:
        limits: List[Optional[Tuple[int, int]]] = [None] * dimensions

        for dp in data:
            for i in range(dimensions):
                # find lowest and highest value in all datapoints for every
                # dimension
                if limits[i] is None:
                    limits[i] = (dp.v_data[i], dp.v_data[i])
                else:
                    if dp.v_data[i] < limits[i][0]:
                        limits[i] = (dp.v_data[i], limits[i][1])
                    if dp.v_data[i] > limits[i][1]:
                        limits[i] = (limits[i][0], dp.v_data[i])

        return limits

    @staticmethod
    def k_means(K: int, data: List['DataPoint'], dimensions: int) -> Label:
        limits = DataPoint.calculate_limits(data, dimensions)

        clusters: List[Cluster] = [Cluster(dimensions, limits) for _ in range(K)]
        cumulative_displacement: int = None

        while cumulative_displacement is None or cumulative_displacement > 1:
            cumulative_displacement = 0
            for dp in data:
                dp.bind_to_closest_cluster(clusters)
            for c in clusters:
                cumulative_displacement += c.update_centroid()

    ## Constructs a new class of DataPoint with a label and vector data
    #
    # @param Label to use for self
    # @param Vector data associated with label
    def __init__(self, label: Label, v_data: Vector):
        self.label: Label = label
        self.v_data: Vector = v_data
        self.cluster: Cluster = None

    ## Computes the euclidian distance between self and another datapoint
    #
    # @param DataPoint other to compute the distance with
    def compute_distance(self, other: 'DataPoint') -> float:
        if isinstance(other, DataPoint):
            return (self.v_data - other.v_data).norm()
        elif isinstance(other, Cluster):
            return (self.v_data - other.centroid).norm()

    def bind_to_closest_cluster(self, clusters: List[Cluster]):
        closest_cluster: Cluster = clusters[0]

        for c in clusters[1:]:
            if self.compute_distance(c) < self.compute_distance(closest_cluster):
                closest_cluster = c

        self.cluster = closest_cluster
        closest_cluster.accumulator += self.v_data
        closest_cluster.point_count += 1

    ## Computes the most plausible label for self using k nearest neighbours
    #
    # @param int K amount of nearest neighbours to use as reference
    # @param List[DataPoint] reference data to compute nearest neighbours of
    def k_nearest_neighbours(self, K: int, data: List['DataPoint']) -> Label:
        neighbors_with_distance = []

        for other_dp in data:
            neighbors_with_distance.append((self.compute_distance(other_dp), other_dp))
            
        neighbors_with_distance.sort(key=lambda tuple: tuple[0], reverse=False)

        return DataPoint.get_dominant_label([tup[1].label for tup in neighbors_with_distance[:K]])

    def __repr__(self):
        return str(self.label) + ": " + str(self.v_data)

def main(k, datapoints):
    DataPoint.k_means(K, datapoints, 7)

    clusters_dict = { }
    cum_distance = 0

    for dp in datapoints:
        cum_distance += ((dp.v_data - dp.cluster.centroid) ** Vector(2, 2)).norm()

        if dp.cluster not in clusters_dict.keys():
            clusters_dict[dp.cluster] = 1
        else:
            clusters_dict[dp.cluster] += 1

    print(clusters_dict)

    return cum_distance, clusters_dict

if __name__ == "__main__":
    xdat: List[int] = [ ]
    ydat: List[int] = [ ]
    
    pool = ThreadPool(processes=8)
    promises: Dict[int, List[ApplyResult]] = { }
    datapoints = DataPoint.from_dataset(dataset.data, dataset.labels)

    for K in range(1, 10):
        xdat.append(K)

        promises[K] = []

        for i in range(10):
            promises[K].append(pool.apply_async(main, (K, copy.deepcopy(datapoints))))

    for K, l in promises.items():
        cum_measurements: float = 0.0

        for p in l:
            cum_measurements, clusters_dict = p.get()

        ydat.append(cum_measurements / len(l))

    plt.plot(np.array(xdat), np.array(ydat))
    plt.ylabel("Average distance")
    plt.xlabel("K")
    plt.show()