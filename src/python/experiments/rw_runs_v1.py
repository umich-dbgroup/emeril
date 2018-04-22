#!/usr/bin/env python
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
    parser.add_argument('--pred_mode', type=str, required=True)
    parser.add_argument('--pred_dep_percents', type=str, required=True)
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--slack_percent', type=float, default=0.2)
    parser.add_argument('--random_data', action='store_true', default=False)
    parser.add_argument('--minos_options', type=str)
    parser.add_argument('--system', type=str, default="emerilRandom")
    parser.add_argument('--max_preds', type=int, default=2)
    parser.add_argument('--mip_solver_timeout', type=int, default=None)
    parser.add_argument('--use_meta_cache', action='store_true', default=False)
    parser.add_argument('--solver', type=str, default='minos')
    parser.add_argument('--add_timeout_to_meta_name', action='store_true', default=False)
    parser.add_argument('--rw_data_mode', type=str, default="rw1")
    parser.add_argument('--use_code_sim', action='store_true', default=False)
    parser.add_argument('--run_id', type=int, default=None)
    args = parser.parse_args()

    pred_dep_percents = [float(x.strip()) for x in args.pred_dep_percents.split(',')]
    if args.minos_options == "dynamic":
        minos_options = args.minos_options
    elif args.minos_options:
        minos_options = {x.split('=')[0]: x.split('=')[1] for x in args.minos_options.split(',')}
    else:
        minos_options = None

    print "Start: {}".format(dt.datetime.now())
    start = timer()

    run_ids = xrange(0, args.num_runs) if args.run_id is None else [args.run_id]

    for dataset_name in args.datasets.split(','):
        for pred_dep_percent in pred_dep_percents:
            for i in run_ids:
                print "\n{} - run={}, dataset={}, pdp={}" \
                    .format(dt.datetime.now(), i, dataset_name, pred_dep_percent)
                sorting_seed = RANDOM_SEED + i
                data_seed = sorting_seed if args.random_data else RANDOM_SEED

                meta = test_real_world_v1(dataset_name, data_seed, pred_dep_percent,
                                          print_details=False,
                                          ignore_meta_cache=(not args.use_meta_cache),
                                          sorting_random_seed=sorting_seed,
                                          slack_percent=args.slack_percent,
                                          minos_options=minos_options,
                                          system=args.system, max_preds=args.max_preds,
                                          mip_solver_timeout=args.mip_solver_timeout,
                                          pred_mode=args.pred_mode,
                                          solver=args.solver,
                                          add_timeout_to_meta_name=args.add_timeout_to_meta_name,
                                          rw_data_mode=args.rw_data_mode,
                                          use_code_sim=args.use_code_sim)
                if meta['mip_statuses'][-1] == 'termcode6':
                    print " - status termcode6"
                elif meta['mip_statuses'][-1] == 'termcode11':
                    print " - status termcode11"
                print "RESULT: status={}, found={}, valid={}, exact={}"\
                    .format(meta['mip_statuses'][-1], meta['solution_found'],
                            meta['is_valid_solution'], meta['exact_solution'])

    print "\n\n\nEnd: {}".format(dt.datetime.now())
    print "Runtime: {}".format(timer() - start)


if __name__ == '__main__':
    main()
