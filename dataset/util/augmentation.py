import random
import numpy as np
from util.seed import set_seed

set_seed()


def mutate_sequence(seq_list, mutation_rate, amino_acids):
    mutation_mask = np.random.rand(len(seq_list)) < mutation_rate
    return [random.choice(amino_acids) if mutate else aa for mutate, aa in zip(mutation_mask, seq_list)]


def insertion_sequence(seq_list, insertion_rate, amino_acids):
    num_insertions = np.random.binomial(len(seq_list), insertion_rate)
    for _ in range(num_insertions):
        insert_idx = np.random.randint(0, len(seq_list))
        seq_list.insert(insert_idx, random.choice(amino_acids))
    return seq_list


def deletion_sequence(seq_list, deletion_rate):
    num_deletions = np.random.binomial(len(seq_list), deletion_rate)
    for _ in range(num_deletions):
        if len(seq_list) > 1:
            delete_idx = np.random.randint(0, len(seq_list))
            del seq_list[delete_idx]
    return seq_list

def augment_sequence(seq, num_fragments=6, mutation_rate=0.5, insertion_rate=0.5, deletion_rate=0.5, multi_step=1):
    seq_list = list(seq)

    fragment_points = sorted(random.sample(range(1, len(seq_list)), min(num_fragments - 1, len(seq_list) - 1)))
    fragments = [seq_list[i:j] for i, j in zip([0] + fragment_points, fragment_points + [None])]

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    for _ in range(multi_step):
        for i in range(len(fragments)):
            fragment = fragments[i]
            if i % 3 == 0:
                fragments[i] = mutate_sequence(fragment, mutation_rate, amino_acids)
            elif i % 3 == 1:
                fragments[i] = insertion_sequence(fragment, insertion_rate, amino_acids)
            else:
                fragments[i] = deletion_sequence(fragment, deletion_rate)


    return ''.join(sum(fragments, []))




