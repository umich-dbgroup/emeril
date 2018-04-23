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
sys.path.append("/home/ec2-user/emeril/src/python")

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


def get_pred_dep_results(num_rows, num_columns, corr, pred_dep_percents,
                         num_runs, slack_percent, random_data, system,
                         minos_options, max_preds, syn_data_mode, num_preds,
                         use_meta_cache, mip_solver_timeout, solver,
                         add_timeout_to_meta_name, use_code_sim, run_id):
    results = dict(
        found=defaultdict(list),
        valid=defaultdict(list),
        exact=defaultdict(list),
    )
    run_ids = xrange(0, num_runs) if run_id is None else [run_id]
    for pred_dep_percent in pred_dep_percents:
        for i in run_ids:
            print "\n{} - run={}, num_rows={}, num_columns={}, corr={}, num_preds={}, pdp={}" \
                .format(dt.datetime.now(), i, num_rows, num_columns, corr, num_preds, pred_dep_percent)
            sorting_seed = RANDOM_SEED + i
            data_seed = sorting_seed if random_data else RANDOM_SEED
            meta = test_syn_data_v2(num_rows, num_columns, corr, data_seed, pred_dep_percent,
                                    print_details=False,
                                    ignore_meta_cache=(not use_meta_cache),
                                    sorting_random_seed=sorting_seed,
                                    slack_percent=slack_percent,
                                    minos_options=minos_options,
                                    system=system, max_preds=max_preds,
                                    syn_data_mode=syn_data_mode,
                                    num_preds=num_preds,
                                    mip_solver_timeout=mip_solver_timeout,
                                    solver=solver,
                                    add_timeout_to_meta_name=add_timeout_to_meta_name,
                                    use_code_sim=use_code_sim
                                    )
            results['found'][pred_dep_percent].append(1 if meta['solution_found'] else 0)
            results['valid'][pred_dep_percent].append(1 if meta['is_valid_solution'] else 0)
            results['exact'][pred_dep_percent].append(1 if meta['exact_solution'] else 0)
            if meta['mip_statuses'][-1] == 'termcode6':
                print " - status termcode6"
            elif meta['mip_statuses'][-1] == 'termcode11':
                print " - status termcode11"
            print "RESULT: status={}, found={}, valid={}, exact={}"\
                .format(meta['mip_statuses'][-1], meta['solution_found'],
                        meta['is_valid_solution'], meta['exact_solution'])
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--pred_dep_percents', type=str, required=True)
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--slack_percent', type=float, default=0.2)
    parser.add_argument('--random_data', action='store_true', default=False)
    parser.add_argument('--minos_options', type=str)
    parser.add_argument('--system', type=str, default="emerilRandom")
    parser.add_argument('--max_preds', type=int, default=2)
    parser.add_argument('--mip_solver_timeout', type=int, default=None)
    parser.add_argument('--corrs', type=str, default=','.join(map(str, np.arange(0.0, 1.1, 0.1))))
    parser.add_argument('--rows', type=str, default=','.join(map(str, range(100000, 1000001, 100000))))
    parser.add_argument('--columns', type=str, default=','.join(map(str, range(1000, 10001, 1000))))
    parser.add_argument('--syn_data_mode', type=str, default="v1")
    parser.add_argument('--preds', type=str, default=','.join(map(str, range(1000, 10001, 1000))))
    parser.add_argument('--use_meta_cache', action='store_true', default=False)
    parser.add_argument('--solver', type=str, default='minos')
    parser.add_argument('--add_timeout_to_meta_name', action='store_true', default=False)
    parser.add_argument('--use_code_sim', action='store_true', default=False)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--default_rows', type=int, default=50000)
    parser.add_argument('--default_cols', type=int, default=100)
    parser.add_argument('--default_corr', type=float, default=0.5)
    parser.add_argument('--default_preds', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'preds' and args.syn_data_mode not in ('v2', 'v2b', 'v2c'):
        raise Exception("preds mode needs syndata mode")

    pred_dep_percents = [float(x.strip()) for x in args.pred_dep_percents.split(',')]
    if args.minos_options == "dynamic":
        minos_options = args.minos_options
    elif args.minos_options:
        minos_options = {x.split('=')[0]: x.split('=')[1] for x in args.minos_options.split(',')}
    else:
        minos_options = None

    default_rows = args.default_rows
    default_cols = args.default_cols
    default_corr = args.default_corr
    default_preds = args.default_preds
    corr_vals = map(float, args.corrs.split(','))
    row_vals = map(int, args.rows.split(','))
    col_vals = map(int, args.columns.split(','))
    pred_vals = map(int, args.preds.split(','))

    print "Start: {}".format(dt.datetime.now())
    start = timer()

    results = defaultdict(list)
    if args.mode == 'corr':
        num_rows = default_rows
        num_columns = default_cols
        num_preds = default_preds
        for i, corr in enumerate(corr_vals):
            results[corr] = get_pred_dep_results(num_rows, num_columns, corr,
                                                 pred_dep_percents, args.num_runs,
                                                 args.slack_percent, args.random_data,
                                                 args.system, minos_options,
                                                 args.max_preds, args.syn_data_mode, num_preds,
                                                 args.use_meta_cache, args.mip_solver_timeout,
                                                 args.solver, args.add_timeout_to_meta_name,
                                                 args.use_code_sim, args.run_id)

    elif args.mode == 'rows':
        num_columns = default_cols
        corr = default_corr
        num_preds = default_preds
        for i, num_rows in enumerate(row_vals):
            results[num_rows] = get_pred_dep_results(num_rows, num_columns, corr,
                                                     pred_dep_percents, args.num_runs,
                                                     args.slack_percent, args.random_data,
                                                     args.system, minos_options,
                                                     args.max_preds, args.syn_data_mode, num_preds,
                                                     args.use_meta_cache, args.mip_solver_timeout,
                                                     args.solver, args.add_timeout_to_meta_name,
                                                     args.use_code_sim, args.run_id)

    elif args.mode == 'columns':
        num_rows = default_rows
        corr = default_corr
        num_preds = default_preds
        for i, num_columns in enumerate(col_vals):
            results[num_columns] = get_pred_dep_results(num_rows, num_columns, corr,
                                                        pred_dep_percents, args.num_runs,
                                                        args.slack_percent, args.random_data,
                                                        args.system, minos_options,
                                                        args.max_preds, args.syn_data_mode, num_preds,
                                                        args.use_meta_cache, args.mip_solver_timeout,
                                                        args.solver, args.add_timeout_to_meta_name,
                                                        args.use_code_sim, args.run_id)

    elif args.mode == 'preds':
        num_rows = default_rows
        num_columns = default_cols
        corr = default_corr
        for i, num_preds in enumerate(pred_vals):
            results[num_columns] = get_pred_dep_results(num_rows, num_columns, corr,
                                                        pred_dep_percents, args.num_runs,
                                                        args.slack_percent, args.random_data,
                                                        args.system, minos_options,
                                                        args.max_preds, args.syn_data_mode, num_preds,
                                                        args.use_meta_cache, args.mip_solver_timeout,
                                                        args.solver, args.add_timeout_to_meta_name,
                                                        args.use_code_sim, args.run_id)


    # processing results
    print "Keys: {}".format(', '.join(['{:.2f}'.format(x) for x in sorted(results.keys())]))
    for key in sorted(results.keys()):
        found = [np.mean(results[key]['found'][pdp]) for pdp in pred_dep_percents]
        valid = [np.mean(results[key]['valid'][pdp]) for pdp in pred_dep_percents]
        exact = [np.mean(results[key]['exact'][pdp]) for pdp in pred_dep_percents]
        output = found + valid + exact
        print '\t'.join(['{:.3f}'.format(x) for x in output])


    print "\n\n\nEnd: {}".format(dt.datetime.now())
    print "Runtime: {}".format(timer() - start)


if __name__ == '__main__':
    main()
