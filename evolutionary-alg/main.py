## @file main.py
#
# @details
# Excercise 6.1, card problem

from math import log, ceil
import random

## Product summation
#
# @details
# Multiplies all members in a list
def prod(lst):
    accumulator = 1
    for v in lst:
        accumulator *= v
    return accumulator

class Genotype:
    def __init__(self, sum=None, product=None):
        if sum is None or product is None:
            assert sum is product, "If either sum or product is None, then both should be none"

            cards = [v for v in range(1, 11)]
            random.shuffle(cards)
            random_index = ceil(random.random() * 10)
        else:
            assert len(sum) + len(product) == 10, "The sum of the length of sum and product should be exactly 10"
            assert sorted(sum + product) == [v for v in range(1, 11)], "All numbers from the range [1, 10] should be included in sum and product, note that 10 is included"
        
        # Everything before random_index is put into self.sum, the remainder is
        # put in self.product
        self.sum = cards[:random_index-1] if sum is None else sum
        self.product = cards[random_index:] if product is None else product

    
    ## Fitness function
    #
    # @details Fitness is calculated as (LaTeX formula):
    #
    # |\sum_{i=1}^{x}{\frac{T_\Sigma - \sum{G_\Sigma}}{G_{\Sigma i}}}|
    # +
    # |\sum_{j=1}^{y}{log_G_{\Pi j}(\frac{\Pi G_\Pi}{T_{\Pi}})}|
    #
    # Please evaluate the above formula in LaTeX, the easiest option would be to
    # use an online tool like https://www.codecogs.com/latex/eqneditor.php
    #
    # Clarification of the formula:
    #
    # x is the amount of items in self.sum and y is the amount of items in
    # self.product. i and j are the indices of the items in the lists.
    # T_sigma is the target_sum which is 36 for this problem. T_pi is the
    # target_product, which is 360 for this problem.
    # G_sigma represents self.sum and G_pi represents self.product. When G_sigma
    # with subscript i is written, that is analogous to self.sum[i]. The same
    # counts for G_pi and self.product.
    #
    # @example
    # assert Genotype([2, 7, 8, 9, 10], [1, 3, 4, 5, 6]).fitness() == 0
    def fitness(self):
        # Juicy one liners
        target_sum = 36
        sum_fitness = abs(sum([(target_sum - sum(self.sum)) / v for v in self.sum]))

        target_prod = 360

        # log(x, 1) raises a ZeroDivisionError because log(1) == 0. So 1 should
        # be removed from self.product if it is in there. This does not affect
        # the fitness because 1 does not contribute to the fitness for the
        # product component.
        tmp_product = self.product[:]
        if 1 in self.product:
            tmp_product.remove(1) 

        prod_fitness = abs(sum([log(prod(self.product)/target_prod, v) for v in tmp_product]))

        return sum_fitness + prod_fitness



if __name__ == "__main__":
    print(Genotype([2, 7, 8, 9, 10], [1, 3, 4, 5, 6]).fitness())