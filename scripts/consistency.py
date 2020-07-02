#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:30:18 2019

@author: masfaraud
"""

import dectree

dt = dectree.RegularDecisionTree([3, -1, 2, 4])

while not dt.finished:
    print(dt.current_node)
    dt.NextNode(True)