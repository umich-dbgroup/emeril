#!/usr/bin/env python
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


def gen_data(num_rows, num_columns, corr, num_runs, syn_data_mode, num_preds, n_processes, run_id):
    run_ids = xrange(0, num_runs) if run_id is None else [run_id]
    for i in run_ids:
        print "{} - run={}, num_rows={}, num_columns={}, corr={}, num_preds={}" \
            .format(dt.datetime.now(), i, num_rows, num_columns, corr, num_preds)
        random_seed = RANDOM_SEED + i
        get_syn_data_and_pred_pair_meta(num_rows, num_columns, num_preds, corr,
                                        random_seed, "emerilRandom", syn_data_mode,
                                        print_details=False, min_print_details=True,
                                        n_processes=n_processes, skip_loading=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--num_runs', type=int, required=True)
    parser.add_argument('--corrs', type=str, default=','.join(map(str, np.arange(0.0, 1.1, 0.1))))
    parser.add_argument('--rows', type=str, default=','.join(map(str, range(100000, 1000001, 100000))))
    parser.add_argument('--columns', type=str, default=','.join(map(str, range(1000, 10001, 1000))))
    parser.add_argument('--syn_data_mode', type=str, default="v1")
    parser.add_argument('--preds', type=str, default=','.join(map(str, range(1000, 10001, 1000))))
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--default_rows', type=int, default=50000)
    parser.add_argument('--default_cols', type=int, default=100)
    parser.add_argument('--default_corr', type=float, default=0.5)
    parser.add_argument('--default_preds', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'preds' and args.syn_data_mode != 'v2':
        raise Exception("preds mode needs syndata mode")

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

    if args.mode == 'corr':
        num_rows = default_rows
        num_columns = default_cols
        num_preds = default_preds
        for i, corr in enumerate(corr_vals):
            gen_data(num_rows, num_columns, corr, args.num_runs,
                     args.syn_data_mode, num_preds, args.n_processes, args.run_id)

    elif args.mode == 'rows':
        num_columns = default_cols
        corr = default_corr
        num_preds = default_preds
        for i, num_rows in enumerate(row_vals):
            gen_data(num_rows, num_columns, corr, args.num_runs,
                     args.syn_data_mode, num_preds, args.n_processes, args.run_id)

    elif args.mode == 'columns':
        num_rows = default_rows
        corr = default_corr
        num_preds = default_preds
        for i, num_columns in enumerate(col_vals):
            gen_data(num_rows, num_columns, corr, args.num_runs,
                     args.syn_data_mode, num_preds, args.n_processes, args.run_id)

    elif args.mode == 'preds':
        num_rows = default_rows
        num_columns = default_cols
        corr = default_corr
        for i, num_preds in enumerate(pred_vals):
            gen_data(num_rows, num_columns, corr, args.num_runs,
                     args.syn_data_mode, num_preds, args.n_processes, args.run_id)

    print "\n\n\nEnd: {}".format(dt.datetime.now())
    print "Runtime: {}".format(timer() - start)


if __name__ == '__main__':
    main()
