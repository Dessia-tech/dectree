#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Barrel file for dectree.
"""

import pkg_resources

__version__ = pkg_resources.require("dectree")[0].version
from .dectree import *

