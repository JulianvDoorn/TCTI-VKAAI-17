## @file main.py
#
# @details
# Excercise 6.1, card problem

from math import log, ceil, floor
import random
from typing import List, Tuple
from copy import deepcopy
from enum import IntEnum

## Product summation
#
# @details
# Multiplies all members in a list
def prod(lst):
    accumulator = 1
    for v in lst:
        accumulator *= v
    return accumulator

GenotypeBit = IntEnum("GenotypeBit", "Sum Product")

class Genotype:
    @staticmethod
    def get_shuffled_cards():
        cards = [v for v in range(1, 11)]
        random.shuffle(cards)
        return cards

    def __init__(self, sum: List[int] = None, product: List[int] = None):
        if sum is None or product is None:
            assert sum is product, "If either sum or product is None, then both should be none"

            cards = Genotype.get_shuffled_cards()
            random_index = ceil(random.random() * 10)
        else:
            assert len(sum) + len(product) == 10, "The sum of the length of sum and product should be exactly 10"
            assert sorted(sum + product) == [v for v in range(1, 11)], "All numbers from the range [1, 10] should be included in sum and product, note that 10 is included"
        
        # Everything before random_index is put into self.sum, the remainder is
        # put in self.product
        self.sum = cards[:random_index-1] if sum is None else sum
        self.product = cards[random_index:] if product is None else product

    ## Constructs a Genotype from a bit encoded vector
    #
    # @details
    # Manual constructions is quite verbose, it should be used primarily in
    # combination with Genotype.to_vector()
    #
    # @example
    # Genotype.from_vector((
    #     GenotypeBit.Sum,
    #     GenotypeBit.Product,
    #     GenotypeBit.Sum,
    #     GenotypeBit.Product,
    #     GenotypeBit.Sum,
    #     GenotypeBit.Product,
    #     GenotypeBit.Sum,
    #     GenotypeBit.Product,
    #     GenotypeBit.Sum,
    #     GenotypeBit.Product,
    # ))
    @classmethod
    def from_vector(cls, vec):
        sum = []
        product = []

        assert len(vec) == 10, "len(vec) should exactly equal 10"

        for i, v in enumerate(vec):
            assert v in list(map(int, GenotypeBit)), "vec should only contain enum values from GenotypeBit"


            if v == 1:
                sum.append(i + 1)
            else:
                product.append(i + 1)

        return cls(sum, product)

    ## Retrieves the genotype as a single tuple, bit encoded
    #
    # @details
    # Encoding goes as following: every bit represents a value in the range
    # [1, 10], the index of said bit determines the classification (as a term
    # or as a factor). If bit[0] == GenotypeBit.Sum, 1 belongs to the list
    # that is summarized. And if bit[1] == GenotypeBit.Product then 2 belongs
    # to the list that is multiplied. The index of the given bit +1 is the value
    # it represents.
    def to_vector(self) -> Tuple[int, ...]:
        lst = [GenotypeBit.Product] * 10

        for v in self.sum:
            lst[v - 1] = GenotypeBit.Sum

        return (*lst,)

    def get_value(self):
        return (sum(self.sum), prod(self.product))

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
        if 1 in tmp_product:
            tmp_product.remove(1) 

        prod_fitness = abs(sum([log(prod(self.product)/target_prod, v) for v in tmp_product]))

        return (sum_fitness + prod_fitness)**2

    def create_child(self, partner):
        _self = self.to_vector()
        partner = partner.to_vector()
        half = int(len(_self) / 2)
        child = _self[:half] + partner[half:]
        return Genotype.from_vector(child)

    def __repr__(self):
        return "Sum fragment: %s Product fragment: %s" % (str(self.sum), str(self.product))

## A generation holds a pool of genotypes with 
class Generation:
    def __init__(self, genotypes: List[Genotype]):
        pass

class EvolutionaryOperators:
    ## Modulo, except it has a provided range
    @staticmethod
    def scoped_mod(v, min, max):
        return (v - min) % max

    ## Flips a value on a complementary basis
    #
    # @details
    # Calculates the complementary value within the range [min, max], note that
    # max is included
    @staticmethod
    def complementary_flip(v, min, max):
        n = max - min
        scoped_v = EvolutionaryOperators.scoped_mod(v, min, max)
        return n - scoped_v + min

    ## Inverts all values, on a complementary basis
    #
    # @details
    # min and max is the range the bits of the genotype belong in. These must
    # be set accordingly in order to find the n'th complement for each bit. 
    @staticmethod
    def flip_genotype(min, max):
        def f(G):
            G = G.to_vector()
            lst = []
            for v in G:
                lst.append(EvolutionaryOperators.complementary_flip(v, min, max))
            return Genotype.from_vector((*lst,))

        return f

    @staticmethod
    def flip_random_gene(min, max):
        def f(G):
            G = G.to_vector()
            i = floor(random.random() * len(G))
            # Make new tuple since tuples are immutable
            return Genotype.from_vector(G[:i] + (EvolutionaryOperators.complementary_flip(G[i], min, max),) + G[i + 1:])

        return f
    
    @staticmethod
    def swap(min, max):
        def f(G):
            G = G.to_vector()
            lhs_i = floor(random.random() * len(G))
            rhs_i = floor(random.random() * len(G))
            while lhs_i == rhs_i or lhs_i > rhs_i:
                if lhs_i == rhs_i:
                    rhs_i = (rhs_i + 1) % len(G)
                if lhs_i > rhs_i:
                    lhs_i, rhs_i = rhs_i, lhs_i
            # Make new tuple since tuples are immutable
            return Genotype.from_vector(G[:lhs_i] + (G[rhs_i],) + G[lhs_i + 1:rhs_i] + (G[lhs_i],) + G[rhs_i + 1:])

        return f


def evolve(population, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [(G.fitness(), G) for G in population]
    graded = [t[1] for t in sorted(graded, key=lambda t: t[0])]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]

    # Promote genetic diversity
    for G in graded[retain_length:]:
        if random_select > random.random():
            parents.append(G)

    desired_length = len(population) - len(parents)
    children = []
    while len(children) < desired_length:
        # With respect to gender diversity, no male and female terms are used
        partner1 = random.choice(parents)
        partner2 = random.choice(parents)
        if partner1 != partner2:
            children.append(partner1.create_child(partner2))

    for i, G in enumerate(children):
        # Possible mutation: swapping cards
        if mutate > random.random():
            swap_mutate = EvolutionaryOperators.swap(GenotypeBit.Sum, GenotypeBit.Product)
            children[i] = swap_mutate(G)

        # Another possible mutation: flipping a gene
        if mutate > random.random():
            swap_mutate = EvolutionaryOperators.flip_random_gene(GenotypeBit.Sum, GenotypeBit.Product)
            children[i] = swap_mutate(G)

        # Another possible mutation: flipping a genotype
        if mutate > random.random():
            swap_mutate = EvolutionaryOperators.flip_genotype(GenotypeBit.Sum, GenotypeBit.Product)
            children[i] = swap_mutate(G)

    parents.extend(children)
    return parents

def grade_population(population, top_percentage=0.1):
    fraction = ceil(top_percentage*len(population))
    return sum([G.fitness() for G in population[:fraction]]) / (fraction - 1)

if __name__ == "__main__":
    population = [Genotype() for _ in range(0, 1000)]
    fitness_history = [grade_population(population)]

    for _ in range(1000):
        population = evolve(population)
        score = grade_population(population)
        fitness_history.append(score)

    possible_combinations = set()

    for G in population:
        if G.get_value() == (36, 360):
            possible_combinations.add(G.to_vector())

    if possible_combinations != set():
        print("Found possible combinations that evaluate to (36, 360):")
        for v in possible_combinations:
            print(Genotype.from_vector(v))

    print("Top 10 of population:")
    population = sorted(population, key=lambda G: G.fitness())
    for G in population[:10]:
        print(G)