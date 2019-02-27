#!/bin/env python3

import numpy as np
import random
import dataset
from vector import Vector
from typing import List, Optional, TypeVar, AnyStr, T, Tuple, Dict
from multiprocessing.pool import ThreadPool, ApplyResult
import matplotlib.pyplot as plt
import math

## Typedef for Label, bound with any string-like object
#
# @details Used by DataPoint to label points
Label = TypeVar("Label", bound=AnyStr)

class Cluster:
    last_cluster_id: int = 0

    @staticmethod
    def pick_random_point(points: List['DataPoint']):
        return random.choice(points)

    @staticmethod
    def generate_random_vector(dimensions, limits):
        return Vector(*[random.uniform(*(limits[i])) for i in range(dimensions)])

    def __init__(self, dimensions, **kwargs):
        self.id = Cluster.last_cluster_id + 1
        Cluster.last_cluster_id += 1

        if "limits" in kwargs.keys() and "random_choice" in kwargs.keys():
            raise Exception("Limits kwarg cannot be used simultaneously with random_choice")
        elif "limits" in kwargs.keys():
            self.centroid = Cluster.generate_random_vector(dimensions, kwargs["limits"])
        elif "random_choice" in kwargs.keys():
            self.centroid = Cluster.pick_random_point(kwargs["random_choice"]).v_data
        else:
            raise Exception("Kwarg missing: limits or random_choice")

        self.accumulator: Vector = Vector(*([0] * dimensions))
        self.point_count = 0
        self.centroid_history = [self.centroid]
        self.members = []

    ## Updates the centroid using the assigned member points
    #
    # @details
    def update_centroid(self):
        if self.point_count != 0:
            self.centroid = Vector(*(a / self.point_count for a in self.accumulator))
        self.accumulator = Vector(*([0] * len(self.accumulator.values)))
        self.point_count = 0
        self.centroid_history.append(self.centroid)
        self.members = []
        return (self.centroid_history[-2] - self.centroid).norm()

    ## Appends the provided datapoint to self.members
    def bind_datapoint(self, datapoint):
        self.members.append(datapoint)

    def __repr__(self):
        return "Cluster " + str(self.id)

## Generic DataPoint class 
#
# @details DataPoint is suitable for different machine learning applications. It
#          has the ability to run the k means algorithms on input data.
class DataPoint:
    ## Converts a List[T] of any applicable T to a List[DataPoint]
    #
    # @details Takes a List[T] dataset and an optional List[Label]. For each
    #          index i, a datapoint from dataset and a label from labels are
    #          bound to eachother.
    #
    # @param List[T] dataset to read from @param Optional[List[Label]] labels to
    #        bind with dataset at index i
    @staticmethod
    def from_dataset(dataset: List[T], labels: Optional[List[Label]]=None) -> List['DataPoint']:
        datapoints: List[DataPoint] = [ ]

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

    ## Finds the lowest and highest value for a given dataset
    #
    # @details For every dimension up to n dimensions the limits are calculated
    #          using List[DataPoint] data. The upper and lower limits are
    #          returned for every dimension in a List[Tuple[LowerLimit,
    #          UpperLimit]]
    #
    # @param List[DataPoint] data @param int N dimensions available in the given
    #        List[DataPoint]
    #
    # @return List[Tuple[LowerLimit, UpperLimit]] Lower and upper limits for
    #         every dimension. Each list element represents the limits for a
    #         dimensions respectively.
    @staticmethod
    def calculate_limits(data: List['DataPoint'], dimensions: int) -> List[Tuple[int, int]]:
        limits: List[Optional[Tuple[int, int]]] = [None] * dimensions

        for dp in data:
            for i in range(dimensions):
                # Find lowest and highest value in all datapoints for every
                # dimension.
                if limits[i] is None:
                    limits[i] = (dp.v_data[i], dp.v_data[i])
                else:
                    if dp.v_data[i] < limits[i][0]:
                        limits[i] = (dp.v_data[i], limits[i][1])
                    if dp.v_data[i] > limits[i][1]:
                        limits[i] = (limits[i][0], dp.v_data[i])

        return limits

    ## Computes and returns the clusters containing references to the associated member datapoints
    #
    # @details Computes k clusters for List[DataPoint] data for N dimensions.
    #          The computation continues until the cumulative displacement of
    #          the clusters' centroids is less or equal than
    #          max_allowed_displacement.
    #
    # @param K clusters to find @param List[DataPoint] dataset to use
    # @param int "dimensions" (kwarg) amount of dimensions in dataset
    # @param float "max_allowed_displacement" (kwarg) continue iterations until
    #        max_allowed_displacement is achieved
    @staticmethod
    def k_means(K: int, data: List['DataPoint'], **kwargs) -> Label:
        assert "dimensions" in kwargs.keys(), "Kwarg dimension missing"

        dimensions = kwargs["dimensions"]
        max_allowed_displacement = kwargs["max_allowed_displacement"] if "max_allowed_displacement" in kwargs.keys() else 0.1

        ready = False
        while not ready:
            if "random_choice" in kwargs.keys():
                clusters: List[Cluster] = [Cluster(dimensions, random_choice=kwargs["random_choice"]) for _ in range(K)]
            elif "limits" in kwargs.keys():
                clusters: List[Cluster] = [Cluster(dimensions, limits=kwargs["limits"]) for _ in range(K)]

            cumulative_displacement: int = None

            while cumulative_displacement is None or cumulative_displacement > max_allowed_displacement:
                cumulative_displacement = 0
                for c in clusters:
                    cumulative_displacement += c.update_centroid()
                for dp in data:
                    dp.bind_to_closest_cluster(clusters)

            # If every cluster has members, the clustering is done properly
            # otherwise the algorithmn runs again.
            ready = True
            for c in clusters:
                if len(c.members) == 0:
                    ready = False

        return clusters

    ## Constructs a new class of DataPoint with a label and vector data
    #
    # @param Label to use for self
    # @param Vector data associated with label
    def __init__(self, label: Label, v_data: Vector):
        self.label: Label = label
        self.v_data: Vector = v_data

    ## Computes the euclidian distance between self and another datapoint
    #
    # @param DataPoint other to compute the distance with
    def compute_distance(self, other: 'DataPoint') -> float:
        if isinstance(other, DataPoint):
            return (self.v_data - other.v_data).norm()
        elif isinstance(other, Cluster):
            return (self.v_data - other.centroid).norm()

    ## Adds a reference to the members list of the closest cluster of List[Clusters] to 'self'
    #
    # @param List[Cluster] clusters to compare with
    def bind_to_closest_cluster(self, clusters: List[Cluster]):
        closest_cluster: Cluster = clusters[0]

        for c in clusters[1:]:
            if self.compute_distance(c) < self.compute_distance(closest_cluster):
                closest_cluster = c

        closest_cluster.bind_datapoint(self)
        closest_cluster.accumulator += self.v_data
        closest_cluster.point_count += 1

    def __repr__(self):
        return str(self.label) + ": " + str(self.v_data)

## Computes the cumulative distance of every datapoint to their corresponding cluster's centroid
#
# @param int K clusters
# @param List[DataPoint] datapoints to compute with
def compute_cum_distance(k: int, datapoints: List[DataPoint]):
    clusters = DataPoint.k_means(K, datapoints, dimensions=7, random_choice=datapoints)
    cum_distance = 0

    for c in clusters:
        for dp in c.members:
            cum_distance += ((dp.v_data - c.centroid) ** Vector(2, 2)).norm()

    return cum_distance

if __name__ == "__main__":
    xdat: List[int] = [ ]
    ydat: List[int] = [ ]
    
    pool = ThreadPool(processes=8)
    promises: Dict[int, List[ApplyResult]] = { }
    datapoints = DataPoint.from_dataset(dataset.data, dataset.labels)

    spawned_threads = 0
    threads_finished = 0

    print("Spawning threads...")

    for K in range(1, 10):
        xdat.append(K)

        promises[K] = []

        for i in range(100):
            promises[K].append(pool.apply_async(compute_cum_distance, (K, datapoints)))
            spawned_threads += 1

    print("Running...")

    for K, l in promises.items():
        print("Waiting for threads: {}/{}".format(threads_finished, spawned_threads))
        cum_measurements: float = 0.0

        for p in l:
            cum_measurements += p.get()
            threads_finished += 1

        ydat.append(cum_measurements / len(l))
        
    plt.plot(np.array(xdat), np.array(ydat))
    plt.ylabel("Average distance")
    plt.xlabel("K")

    # derivative
    deriv = np.diff(np.array(ydat), n=2)
    plt.plot(np.array(xdat[:7]), deriv, color="green")

    for K in range(1, len(deriv)-1):
        if deriv[K] < deriv[K+1]:
            print("Found optimal K:", K+1)
            break

    plt.show()