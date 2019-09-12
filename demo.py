# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件展现如何使用LAC的方法。

Authors: huangdingbang(huangdingbang@baidu.com)
Date:    2019/09/02 21:09:21
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from LAC import LAC


if __name__ == "__main__":
    lac = LAC('infer_model')

    # 对于单个样本输入
    test_data = [u'百度是一家高科技公司']
    result = lac.lac_seg(test_data)
    for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))

    print('#####################################')
    # 对于batch的输入
    test_data = [u'百度是一家高科技公司', u'中山大学是岭南第一学府']
    result = lac.lac_seg(test_data)
    for i, (sent, tags) in enumerate(result):
        result_list = ['(%s, %s)' % (ch, tag) for ch, tag in zip(sent, tags)]
        print(''.join(result_list))

