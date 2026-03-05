# -*- coding: utf-8 -*-
"""BaseDT 数据工具模块。"""

from .dataset import make_npz_dataset, NPZGenerator
from . import npz

__all__ = ['make_npz_dataset', 'NPZGenerator', 'npz']
