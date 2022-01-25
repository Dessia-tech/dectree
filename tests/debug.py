#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:45:28 2018

@author: jezequel
"""
import dectree

dt = dectree.DecisionTree()
node = dt.current_node
dt.SetCurrentNodeNumberPossibilities(3)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(2)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(1)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(2)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(3)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(2)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(4)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
dt.SetCurrentNodeNumberPossibilities(0)
print(node)
print(dt.finished)

node = dt.NextNode(True)
print(node)
print(dt.finished)

#print(dt.current_depth_np_known)
#print(dt.finished)