

import dectree

class Composant:
    def __init__(self, num):
        self.num = num

dt = dectree.DecisionTree()
# vector: [nb_comp_edge1, pos_edge1_comp1, pos_edge1_comp2, ..., nb_comp_edge2, pos_edge2_comp1, ...]
# composants = [comp1, comp2, comp3, ...]
composants = [Composant(i) for i in range(6)]
nb_composants_max = len(composants)
nb_composants_max_per_edge = 3

solutions = []
while not dt.finished:
    valid = True

    # read of the current vector
    if dt.current_depth > 0:

        list_nb_composants = []
        node_composants = []
        current = 0
        for i, node in enumerate(dt.current_node):
            if i == current:
                list_nb_composants.append(node + 1)
                current += node + 2
            else:
                node_composants.append(node)

        if len(set(node_composants)) != len(node_composants):
            valid = False
        nb_composants = sum(list_nb_composants)
        if valid:
            if nb_composants == nb_composants_max and dt.current_depth == nb_composants + len(list_nb_composants):
                dt.SetCurrentNodeNumberPossibilities(0)
            else:
                if dt.current_depth == nb_composants + len(list_nb_composants):
                    if nb_composants_max - nb_composants > nb_composants_max_per_edge:
                        dt.SetCurrentNodeNumberPossibilities(nb_composants_max_per_edge)
                    else:
                        dt.SetCurrentNodeNumberPossibilities(nb_composants_max - nb_composants)
                elif dt.current_depth < nb_composants + len(list_nb_composants):
                    dt.SetCurrentNodeNumberPossibilities(len(composants))
        else:
            dt.SetCurrentNodeNumberPossibilities(0)

        if valid and nb_composants == nb_composants_max and dt.current_depth == nb_composants + len(list_nb_composants):
            # print(valid, dt.current_node)
            # print(list_nb_composants)
            # print(node_composants)
            solutions.append({'nb_composants': nb_composants, 'nb_edges': len(list_nb_composants), 'list_composants': node_composants,
                              'vector': dt.current_node})
            if len(solutions)/10000 == int(len(solutions)/10000):
                print('number of solutions obtain {}'.format(len(solutions)))
    else:
        dt.SetCurrentNodeNumberPossibilities(nb_composants_max_per_edge)

    dt.NextNode(valid)

print('number solution {}'.format(len(solutions)))
print(solutions[0])