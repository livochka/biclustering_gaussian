import sklearn
import numpy as np

def cluster_accuracy(a, b):
    m = sklearn.metrics.confusion_matrix(a, b)
    
    state = {}
    def best_of(left):
        if len(left) == 1:
            return (left, m[-1, left[0]])
        if left not in state:
            i = len(m) - len(left)
            best_score = 0
            best_seq = 0
            for j in left:
                left_j = tuple([v for v in left if v != j])
                seq_j, score_j = best_of(left_j)
                score_j += m[i, j]

                if score_j > best_score:
                    best_score = score_j
                    best_seq = (j, seq_j)

            state[left] = (best_seq, best_score)

        return state[left]

    def unwrap(path):
        if len(path) == 1:
            return path
        inner = unwrap(path[1])
        return [path[0], *inner]

    def mapping(path):
        indexes = sorted(set(a))
        return {indexes[path[i]]: indexes[i] for i in range(len(m))}

    all_left = tuple(range(len(m)))
    best_seq, best_score = best_of(all_left)
    return mapping(unwrap(best_seq)), best_score / len(a)


def map_labels(labels):
    mapping = {v: i for i, v in enumerate(np.unique(labels))}
    labels = np.array([mapping[x] for x in labels])
    return labels
