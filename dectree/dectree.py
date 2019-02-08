#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:53:24 2018

@author: jezequel
"""
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as npy
from copy import copy

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

        # total number of leaves on tree
        self.number_leaves = 1
        for npi in np:
            self.number_leaves *= (npi)

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

    def Progress(self, ndigits=3):
        """
        Compute progress, float between 0 (begin) and 1 (finished) with ndigits rounding
        """
        nll = 0#Number of leaves on the left

        for ind1, pi in enumerate(self.current_node):
            nlli = pi
            for ind2 in range(ind1+1, self.n):
                nlli *= self.np[ind2]
            nll += nlli

        return round(nll/self.number_leaves, ndigits)

    def PlotData(self,
                 valid_nodes,
                 complete_graph_layout=True,
                 plot=False):
        """
        Draws decision tree

        :param valid_nodes: List of tuples that represents nodes to draw
        :param complete_graph_layout: Boolean
        :param plot: Boolean to directly plot or not
        """
        tree_plot_data = []

        nodes = list(set(valid_nodes))
        nodes.sort(key=itemgetter(*range(len(self.np))))
        up_nodes = []
        y_positions = [10*j for j in range(len(self.np)+2)]
        j = 0
        positions = {}
        links = []
        while j <= len(self.np):
            for i, node in enumerate(nodes):
                parent = node[:-1]
                links.append([parent, node])

                if parent not in up_nodes:
                    if up_nodes and not complete_graph_layout:
                        # Add positions to previous parent
                        p = up_nodes[-1]
                        x = PositionParent(p, links, positions)
                        positions[p] = (x, y_positions[j+1])
                        tree_plot_data.append({'type' : 'circle',
                                               'cx' : x,
                                               'cy' : y_positions[j+1],
                                               'r' : 1,
                                               'color' : [0, 0, 0],
                                               'size' : 1,
                                               'dash' : 'none'})
                    up_nodes.append(parent)
                if complete_graph_layout:
                    if len(node) == len(self.np):
                        r = range(len(node) - 1)
                        offset = node[-1]
                    else:
                        r = range(len(node))
                        offset = (npy.prod(self.np[-j:]) - 1)/2
                    start_pos = sum([npy.prod(self.np[k+1:])*node[k] for k in r])
                    x = start_pos + offset

                    positions[node] = (x, y_positions[j])
                    tree_plot_data.append({'type' : 'circle',
                                           'cx' : x,
                                           'cy' : y_positions[j],
                                           'r' : 1,
                                           'color' : [0, 0, 0],
                                           'size' : 1,
                                           'dash' : 'none'})

                else: # Minimal layout graph
                    if len(node) == len(self.np):
                        # Add position to all the leafs
                        x = i
                        positions[node] = (x, y_positions[j])
                        tree_plot_data.append({'type' : 'circle',
                                               'cx' : x,
                                               'cy' : y_positions[j],
                                               'r' : 1,
                                               'color' : [0, 0, 0],
                                               'size' : 1,
                                               'dash' : 'none'})

            if not complete_graph_layout:
                # Add position to the last parent of the line
                p = up_nodes[-1]
                x = PositionParent(p, links, positions)
                positions[p] = (x, y_positions[j+1])
                tree_plot_data.append({'type' : 'circle',
                                       'cx' : x,
                                       'cy' : y_positions[j+1],
                                       'r' : 1,
                                       'color' : [0, 0, 0],
                                       'size' : 1,
                                       'dash' : 'none'})
            j += 1

            nodes = up_nodes
            up_nodes = []

        tree_plot_data.extend(CreateLinks(links, positions))

        if plot:
            Plot(tree_plot_data)

        return tree_plot_data

class DecisionTree:
    """
    Create a general decision tree object
    """
    def __init__(self):
        self.current_node = []
        self._data = {}
        self.finished = False
        self.np = []
        self.current_depth_np_known = False

    def _get_current_depth(self):
        return len(self.current_node)

    current_depth = property(_get_current_depth)

    def _get_data(self):
        
        return self._data

    def _set_data(self):
        msg = ('Should not directly set data attribute.\
               Elements are passed via data arguments of SetCurrentNodePossibilities() method')
        raise RuntimeError(msg)

    data = property(_get_data, _set_data)

    def NextNode(self, current_node_viability):
        """
        Selects next node in decision tree if current one is valid

        :param current_node_viability: boolean
        """
        if (not current_node_viability) or (self.np[self.current_depth] == 0):
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

            self.finished = finished and self.current_depth_np_known

        else:
            if not self.current_depth_np_known:
                raise RuntimeError
                # Going deeper in tree
            self.current_node.append(0)

        ancestors = self.Ancestors()        
        if ancestors is not None:
            _data = copy(self._data)
            self._data = {}
            for node, node_data in _data.items():
                already_visited = self.AlreadyVisited(node)
                ancestor = node in ancestors
                if ancestor or not already_visited:
                    self._data[tuple(node)] = node_data
#            print('node : ', self.current_node, 'ancestors', ancestors)
#            self._data = {tuple(node) : _data[tuple(node)]
#                          for node in ancestors + [self.current_node]}
        return self.current_node


    def SetCurrentNodeNumberPossibilities(self, np_node):
        """
        Set number of nodes under the current node
        :param np_node: an integer representing the number of nodes under the current node
        """
        try:
            self.np[self.current_depth] = np_node
        except IndexError:
            self.np.append(np_node)

        self.current_depth_np_known = True

    def SetCurrentNodeDataPossibilities(self, data_list):
        """
        Set number of nodes under the current node by giving a list of data
        The number of nodes under the current node will be the length of the data array
        :param data_list: a list or tuple of data for each node under
        """
        np_node = len(data_list)
        try:
            self.np[self.current_depth] = np_node
        except IndexError:
            self.np.append(np_node)

        for i, data in enumerate(data_list):
            child = self.current_node + [i]
#            print(child, tuple(child), data)
            self._data[tuple(child)] = data
        self.current_depth_np_known = True

    def AlreadyVisited(self, node):
        booleans = [node[i] < self.current_node[i] for i in range(len(node[:self.current_depth]))]
        if any(booleans):
            return True
        return False

    def Ancestors(self, node=None):
        if node is None:
            node = self.current_node
#        ancestors = [None]*(len(node)-1)
#        for i in range(len(node)-1):
#            ancestors[i] = node[:i+1]
        ancestors = [node[:i+1] for i in range(len(node)-1)]
        if ancestors:
            return ancestors
        return None

def PositionParent(parent, links, positions):
    pos_children = [positions[lk[1]][0] for lk in links if parent in lk]
    pos_parent = (max(pos_children) + min(pos_children))/2
    return pos_parent

def CreateLinks(links, positions):
    link_plot_data = []
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
        link_plot_data.append(element)
    return link_plot_data

def Plot(plot_data):
    """
    TODO Temporary function
    In the future, use a PlotData package
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for element in plot_data:
        if element['type'] == 'circle':
            ax.plot(element['cx'],
                    element['cy'],
                    'o',
                    color=element['color'])

        if element['type'] == 'line':
            ax.plot([element['data'][0], element['data'][2]],
                    [element['data'][1], element['data'][3]],
                    color=element['color'],
                    marker=element['marker'],
                    linewidth=element['stroke_width'])
