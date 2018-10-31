#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:45:28 2018

@author: jezequel
"""
import dectree

dt = dectree.DecisionTree()
node = dt.current_node
dt.SetCurrentNodePossibilities(3)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(2)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(1)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(2)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(3)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(2)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(4)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodePossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
print(node)
print(dt.finished)

#print(dt.current_depth_np_known)
#print(dt.finished)