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
import itertools
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

PARAMS = {
    # default 1.0e-06
    'feasibility_tolerance': ('0.000001', '0.00001', '0.0001', '0.001', '0.01',
                              '0.03', '0.05', '0.07', '0.09', '0.1'),

    # default: 50
    'major_iterations': ('500', ),

    # default 40
    'minor_iterations': ('40', '50', '75', '80', '100', '125', '175',
                         '500', '1000'),

    # default: 3
    'crash_option': ('0', '1', '2', '3'),

    # default 0.1
    # 'crash_tolerance': ('0.1', '0.01', '0.5', '1.0', '0.05'),

    # default: partial
    'completion': ('partial', 'full'),

    # default: 0.0
    'weight_on_linear_objective': ('0.0', '2.0'),

    # default: 10
    'partial_price': ('10', '9', '8', '7'),

    # default: 50
    'superbasics_limit': ('50', '1000'),

    # default: 1.0
    #'penalty_parameter': ('1', '2', '4', '10'),

    # default: 1.0e-06
    #'row_tolerance': ('1.0e-06', '0.1'),

    # default: 2
    #'scale_option': ('1', '2'),
}


# DEBUGGING PARAMS:
# PARAMS = {
#     'feasibility_tolerance': ('0.000001', '0.01'),
#     'major_iterations': ('500', ),
#     'minor_iterations': ('40', '50'),
#     'crash_option': ('0', '1',),
#     'completion': ('partial', 'full'),
#     'weight_on_linear_objective': ('0.0', '2.0'),
#     'partial_price': ('10', '9'),
#     'superbasics_limit': ('50', '1000'),
# }


def _parallel_param_search(pid, queue, param_set_chunk, orig_model_file):
    results = []
    model_file = '/tmp/test-{}.mod'.format(pid)
    processed = 0
    for i, param_set in enumerate(param_set_chunk):
        minos_str = ' '.join(param_set)
        try:
            mip_status, mip_answer_ids = test_minos_options(minos_str, orig_model_file, model_file)
        except Exception as e:
            print "mid={}, run_id={}, pid={}, exception: {}".format(mid, run_id, pid, )
            print "Error with minos_str={}, model_file={}".format(minos_str, model_file)
            continue
        results.append((mip_status, minos_str))
        if mip_answer_ids:
            print "FOUND: status={}, minos_str: {}".format(mip_status, minos_str)
        if (processed + 1) % 50 == 0:
            print "{} - pid={}, processed={}".format(dt.datetime.now(), pid, processed)
        processed += 1
    print("{} - pid={} done; processed={}".format(dt.datetime.now(), pid, processed))
    queue.put(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_model_file', type=str, required=True)
    parser.add_argument('--n_processes', type=int, default=1)
    args = parser.parse_args()
    start = timer()
    num_runs = np.product([len(PARAMS[param]) for param in PARAMS.iterkeys()])
    print("{} - num_runs: {} (~{:.0f}/proc)"\
          .format(dt.datetime.now(), num_runs, float(num_runs) / args.n_processes))
    print("orig_model_file = {}".format(args.orig_model_file))

    # building param sets
    param_strs = []
    for p in PARAMS.iterkeys():
        param_strs.append(['{}={}'.format(p, pv) for pv in PARAMS[p]])
    param_sets = itertools.product(*param_strs)
    param_set_chunks = [
        itertools.islice(p, i, None, args.n_processes)
        for i, p in enumerate(itertools.tee(param_sets, args.n_processes))
    ]

    # processing in parallel
    queue = mp.Queue()
    for pid, param_set_chunk in enumerate(param_set_chunks):
        p = mp.Process(target=_parallel_param_search,
                       args=(pid, queue, param_set_chunk, args.orig_model_file))
        p.Daemon = True
        p.start()
    results = []
    for i in xrange(len(param_set_chunks)):
        results += queue.get()
    queue.close()

    # saving results to disk
    outfile = '/z/dol/emeril/param_search_res-{}.cache'.format(hash(args.orig_model_file))
    print("Output file: {}".format(outfile))
    with open(outfile, "w") as f:
        pickle.dump(results, f, -1)


    print "{} - Done running ({} sec)".format(dt.datetime.now(), timer() - start)


if __name__ == '__main__':
    main()
