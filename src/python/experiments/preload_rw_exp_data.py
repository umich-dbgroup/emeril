#!/usr/bin/env python
##
## Copyright (c) 2018 The Regents of The University of Michigan
## 
## This file is part of Emeril
## (see https://www.github.com/umich-dbgroup/emeril/).
## 
## Licensed to the Apache Software Foundation (ASF) under one
## or more contributor license agreements.  See the NOTICE file
## distributed with this work for additional information
## regarding copyright ownership.  The ASF licenses this file
## to you under the Apache License, Version 2.0 (the
## "License"); you may not use this file except in compliance
## with the License.  You may obtain a copy of the License at
## 
##   http://www.apache.org/licenses/LICENSE-2.0
## 
## Unless required by applicable law or agreed to in writing,
## software distributed under the License is distributed on an
## "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
## KIND, either express or implied.  See the License for the
## specific language governing permissions and limitations
## under the License.
## 
## Author: Dolan Antenucci (dol@umich.edu)
##
import argparse
from collections import defaultdict
import datetime as dt
import sys
sys.path.append("/Users/dol/workspace/emeril/src/python")
sys.path.append("/home/dol/emeril/src/python")

from emeril.utils import *
from emeril.algos import *
from emeril.baselines import *
from emeril.candidate_gen import *
from emeril.evaluation import *
from emeril.find_interesting_fields import *
from emeril.identify_issues import *
from emeril.initialization import *
from emeril.mipsolver import *
from emeril.syndata import *
from emeril.realworld import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--pred_mode', type=str, required=True)
    parser.add_argument('--n_processes', type=int, default=1)
    args = parser.parse_args()

    print "Start: {}".format(dt.datetime.now())
    start = timer()

    for dataset_name in args.datasets.split(','):
        for i in xrange(0, args.num_runs):
            print "{} - run={}, dataset={}".format(dt.datetime.now(), i, dataset_name)
            random_seed = RANDOM_SEED + i
            get_real_world_data_and_pred_pair_meta(dataset_name, random_seed,
                                                   "emerilRandom",
                                                   pred_mode=args.pred_mode,
                                                   print_details=False,
                                                   min_print_details=True,
                                                   n_processes=args.n_processes,
                                                   skip_loading=True)
    print "\n\n\nEnd: {}".format(dt.datetime.now())
    print "Runtime: {}".format(timer() - start)


if __name__ == '__main__':
    main()
