#!/bin/env python3

import dataset
from vector import Vector
from typing import List, Optional, TypeVar, AnyStr, T

## Typedef for Label, bound with any string-like object
#
# @brief Used by DataPoint to label points
Label = TypeVar("Label", bound=AnyStr)

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
            temp = []

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

    ## Constructs a new class of DataPoint with a label and vector data
    #
    # @param Label to use for self
    # @param Vector data associated with label
    def __init__(self, label: Label, v_data: Vector):
        self.label = label
        self.v_data = v_data

    ## Computes the euclidian distance between self and another datapoint
    #
    # @param DataPoint other to compute the distance with
    def compute_distance(self, other: 'DataPoint') -> float:
        return (self.v_data - other.v_data).norm()

    ## Computes the most plausible label for self using k nearest neighbours
    #
    # @param int K amount of nearest neighbours to use as reference
    # @param List[DataPoint] reference data to compute nearest neighbours of
    def k_nearest_neighbours(self, K: int, data: List['DataPoint']) -> Label:
        neighbors_with_distance = []

        for other_dp in data:
            neighbors_with_distance.append((self.compute_distance(other_dp), other_dp))
            
        neighbors_with_distance.sort(key=lambda tuple: tuple[0])

        return DataPoint.get_dominant_label([tup[1].label for tup in neighbors_with_distance[1:K+1]])

    def __repr__(self):
        return str(self.label) + ": " + str(self.v_data)

def main(k):
    datapoints = DataPoint.from_dataset(dataset.data)
    validation_datapoints = DataPoint.from_dataset(dataset.validation_data, dataset.labels)

    estimated_labels: List[DataPoint] = []

    for dp in datapoints:
        estimated_labels.append(dp.k_nearest_neighbours(k, validation_datapoints))

    total_correct = 0
    total_labels = len(estimated_labels)

    for i in range(total_labels):
        if estimated_labels[i] == dataset.labels[i]:
            total_correct += 1

    print("for k: " + str(k))
    print("total correct: " + str(total_correct))
    print("total labels: " + str(total_labels))
    print("percentage: " + str(total_correct / total_labels * 100))

if __name__ == "__main__":
    print("Starting algo")
    main(42)