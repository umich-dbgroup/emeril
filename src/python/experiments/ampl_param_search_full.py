#!/usr/bin/env python
import argparse
from collections import defaultdict
import datetime as dt
import sys
import itertools
import random
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

from ampl_param_search import PARAMS
from ampl_param_search_phase2 import _parallel_param_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()
    datasets = args.datasets.split(',')
    start = timer()

    # 1. building set of jobs
    param_strs = []
    for p in PARAMS.iterkeys():
        param_strs.append(['{}={}'.format(p, pv) for pv in PARAMS[p]])
    param_sets = itertools.product(*param_strs)

    # 2. getting results for each dataset
    for dataset_name in datasets:
        print "\n\n# Starting dataset: {} ({}) #".format(dataset_name, dt.datetime.now())
        # 2a. building set of jobs
        param_set_chunks = [
            itertools.islice(p, i, None, args.n_processes)
            for i, p in enumerate(itertools.tee(param_sets, args.n_processes))
        ]

        # 2b. processing in parallel
        queue = mp.Queue()
        for pid, param_set_chunk in enumerate(param_set_chunks):
            p = mp.Process(target=_parallel_param_search,
                           args=(pid, queue, param_set_chunk, dataset_name, args.num_runs))
            p.Daemon = True
            p.start()
        results = []
        for i in xrange(len(param_set_chunks)):
            results += queue.get()
        queue.close()

        # 2c. sorting and outputting
        print("\n## RESULTS (TOP 1000): ##")
        sorted_results = sorted(results, key=lambda x: -len(x[1]))
        for i, (minos_str, run_ids) in enumerate(sorted_results[0:1000]):
            print "{}. cnt={}, mids={}, minos_str: {}".format(i+1, len(run_ids), run_ids, minos_str)
        print "\n{} - Done running ({} sec)".format(dt.datetime.now(), timer() - start)


if __name__ == '__main__':
    main()
