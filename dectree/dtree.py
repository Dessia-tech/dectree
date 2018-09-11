#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:53:24 2018

@author: jezequel
"""
from functools import reduce
from operator import itemgetter
import matplotlib.pyplot as plt

class RegularDecisionTree:
    """
    Create a regular decision tree object

    :param np: number of possibilities at each depth
    """
    def __init__(self, np):
        self.np = np
        self.n = len(np)
        self._target_node = [0]*self.n
        self.current_depth = 0
        self.finished = False

    def _get_current_node(self):
        return self._target_node[:self.current_depth+1]

    current_node = property(_get_current_node)

    def NextNode(self, current_node_viability):
        """
        Selects next node in decision tree if current one is valid

        :param current_node_viability: boolean
        """
        if current_node_viability:
            self.current_depth += 1

            if self.current_depth == self.n:
                self.current_depth -= 1
                self._target_node[self.current_depth] += 1

        else:
            self._target_node[self.current_depth] += 1

        # Changer les décimales si besoin
        for k in range(self.n-1):
            pk = self._target_node[self.n-1-k]
            if pk >= self.np[self.n-1-k]:
                self._target_node[self.n-1-k] = pk%self.np[self.n-1-k]
                self._target_node[self.n-2-k] += pk//self.np[self.n-1-k]
                self.current_depth = self.n-2-k

        # Return None if finished
        if self.current_node[0] >= self.np[0]:
            self.finished = True
            return None

        return self.current_node

    def NextSortedNode(self, current_node_viability):
        """
        TODO Docstring
        """
        not_sorted = True
        node = self.NextNode(current_node_viability)
        while not_sorted:
            if node is None:
                return node
            if sorted(node) == node:
                not_sorted = False
            else:
                node = self.NextNode(False)
        return node

    def NextUniqueNode(self, current_node_viability):
        """
        TODO Docstring
        """
        not_unique = True
        node = self.NextNode(current_node_viability)
        while not_unique:
            if node is None:
                return node
            if not node[-1] in node[:-1]:
                not_unique = False
            else:
                node = self.NextNode(False)
        return node

    def NextSortedUniqueNode(self, current_node_viability):
        """
        TODO Docstring
        """
        not_unique = True
        node = self.NextSortedNode(current_node_viability)
        while not_unique:

            if node is None:
                return node
            if not node[-1] in node[:-1]:
                not_unique = False
            else:
                node = self.NextSortedNode(False)
        return node

    def DrawTree(self, valid_nodes,
                 complete_graph_layout=True,
                 plot=False):
        """
        Draws decision tree

        :param valid_nodes: List of tuples that represents nodes to draw
        :param complete_graph_layout: Boolean
        :param plot: Boolean to directly plot or not
        """
        if valid_nodes:
            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(111)

            tree_plot_data = []

            nodes = list(set(valid_nodes))
            nodes.sort(key=itemgetter(*range(len(self.np))))
            up_nodes = []
            finished = False
            j = 0
            y_increment = 10
            positions = {}
            links = []
            while not finished:
                for i, node in enumerate(nodes):
                    y = y_increment*j
                    if node:
                        # If node has parent
                        parent = node[:-1]
                        links.append([parent, node])
                    else:
                        # Else, we reached root
                        finished = True
                    if parent not in up_nodes:
                        if up_nodes and not complete_graph_layout:
                            # Add positions to previous parent
                            p = up_nodes[-1]
                            pos_children = [positions[lk[1]][0] for lk in links if p in lk]
                            pos_parent = (max(pos_children) + min(pos_children))/2
                            positions[p] = (pos_parent, y + y_increment)
                        up_nodes.append(parent)
                    if complete_graph_layout:
                        if len(node) == len(self.np):
                            r = range(len(node) - 1)
                            offset = node[-1]
                        else:
                            r = range(len(node))
                            offset = (Prod(self.np[-j:]) - 1)/2
                        start_pos = sum([Prod(self.np[k+1:])*node[k] for k in r])
                        x = start_pos + offset

                        positions[node] = (x, y)
                        point = {'type' : 'circle',
                                 'cx' : x,
                                 'cy' : y,
                                 'r' : 1,
                                 'color' : [0, 0, 0],
                                 'size' : 1,
                                 'dash' : 'none'}
                        tree_plot_data.append(point)
                        if plot:
                            ax.plot(point['cx'],
                                    point['cy'],
                                    color=point['color'])
                    else: # Minimal layout graph
                        if len(node) == len(self.np):
                            # Add position to all the leafs
                            x = i
                            positions[node] = (x, y)
                            point = {'type' : 'circle',
                                     'cx' : x,
                                     'cy' : y,
                                     'r' : 1,
                                     'color' : [0, 0, 0],
                                     'size' : 1,
                                     'dash' : 'none'}
                            tree_plot_data.append(point)
                            if plot:
                                ax.plot(point['cx'],
                                        point['cy'],
                                        color=point['color'])
                if not complete_graph_layout:
                    # Add position to the last parent of the line
                    pos_children = [positions[link[1]][0] for link in links if up_nodes[-1] in link]
                    pos_parent = (max(pos_children) + min(pos_children))/2
                    positions[up_nodes[-1]] = (pos_parent, y+y_increment)
                j += 1

                nodes = up_nodes
                up_nodes = []

            for link in links:
                parent, node = link
                xp, yp = positions[parent]
                xn, yn = positions[node]
                element = {'type' : 'line',
                           'data' : [xp,
                                     yp,
                                     xn,
                                     yn],
                           'color' : [0, 0, 0],
                           'dash' : 'none',
                           'stroke_width' : 1,
                           'marker' : '',
                           'size' : 1}
                tree_plot_data.append(element)

                if plot:
                    ax.plot([element['data'][0], element['data'][2]],
                            [element['data'][1], element['data'][3]],
                            color=element['color'],
                            marker=element['marker'],
                            linewidth=element['stroke_width'])

            return tree_plot_data

        else:
            print('valid_nodes is empty')

class DecisionTree:
    """
    Create a general decision tree object
    """
    def __init__(self):
        self.current_node = []
        self.finished = False
        self.np = []
        self.current_depth_np_known = False

    def _get_current_depth(self):
        return len(self.current_node)

    current_depth = property(_get_current_depth)

    def NextNode(self, current_node_viability):
        """
        Selects next node in decision tree if current one is valid

        :param current_node_viability: boolean
        """
        if (self.np[self.current_depth] == 0)|(not current_node_viability):
            # Node is a leaf | node is not viable
            # In both cases next node as to be searched upwards
            n = self.current_depth

            finished = True
            for i, node in enumerate(self.current_node[::-1]):
                if node != self.np[n-i-1]-1:
                    self.current_node = self.current_node[:n-i]
                    self.current_node[-1] += 1
                    self.current_depth_np_known = False
                    self.np = self.np[:self.current_depth]
                    finished = False
                    break
                if finished:
                    self.finished = True

        else:
            if not self.current_depth_np_known:
                raise RuntimeError
                # Going deeper in tree
            self.current_node.append(0)

        return self.current_node

    def SetCurrentNodePossibilities(self, np_node):
        """
        TODO Docstring
        """
        try:
            self.np[self.current_depth] = np_node
        except IndexError:
            self.np.append(np_node)
        self.current_depth_np_known = True

def Prod(vector):
    """
    Operate products of each elements of given vector

    :param vector: List of numbers
    """
    prod = reduce(lambda x, y: x*y, vector)
    return prod
