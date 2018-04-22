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


def _parallel_param_search(pid, queue, jobs_chunk, dataset_name, num_runs):
    results = defaultdict(list)
    processed = 0
    for mid, minos_str in enumerate(jobs_chunk):
        for run_id in xrange(num_runs):
            mod_fname = '{}-pmode.naive-seed{}-sys.emIndep0-pdp0.001-sort{}-sp0.2.slack0.2.mod'\
                        .format(dataset_name, RANDOM_SEED+run_id, RANDOM_SEED+run_id)
            orig_model_file = os.path.join('/z/dol/emeril/cache/mip_input_files/rw1/', mod_fname)
            model_file = os.path.join('/tmp', mod_fname)
            try:
                mip_status, mip_answer_ids = test_minos_options(minos_str, orig_model_file, model_file)
                #print "DEBUG: done with mip processing: status={}".format(mip_status)
            except Exception as e:
                print "mid={}, run_id={}, pid={}, exception: {}".format(mid, run_id, pid, )
                print "Error with minos_str={}, model_file={}".format(minos_str, model_file)
                continue
            if mip_answer_ids:
                results[minos_str].append(run_id)
            processed += 1
            if (processed) % 500 == 0:
                print "{} - pid={}, num processed: {}".format(dt.datetime.now(), pid, processed)
    print("{} - pid={} done; processed={}".format(dt.datetime.now(), pid, processed))
    queue.put(results.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opts_file', type=str, required=True)
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--n_processes', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()
    datasets = args.datasets.split(',')
    start = timer()

    # 1. get minos_strs
    param_val_counts = defaultdict(lambda: defaultdict(int))
    minos_strs = []
    with open(args.opts_file) as f:
        for line in f:
            minos_strs.append(line.strip())
    num_to_process = len(minos_strs) * args.num_runs
    print("{} - num to process: {} (~{:.0f}/proc)"\
          .format(dt.datetime.now(), num_to_process, float(num_to_process) / args.n_processes))
    print("opts_file: {}".format(args.opts_file))

    # 2. getting results for each dataset
    for dataset_name in datasets:
        print "\n\n# Starting dataset: {} ({}) #".format(dataset_name, dt.datetime.now())
        # 2a. building set of jobs
        jobs_chunks = [
            itertools.islice(p, i, None, args.n_processes)
            for i, p in enumerate(itertools.tee(minos_strs, args.n_processes))
        ]

        # 2b. processing in parallel
        queue = mp.Queue()
        for pid, jobs_chunk in enumerate(jobs_chunks):
            p = mp.Process(target=_parallel_param_search,
                           args=(pid, queue, jobs_chunk, dataset_name, args.num_runs))
            p.Daemon = True
            p.start()
        results = []
        for i in xrange(len(jobs_chunks)):
            results += queue.get()
        queue.close()

        # 2c. sorting and outputting
        print("\n## RESULTS: ##")
        sorted_results = sorted(results, key=lambda x: -len(x[1]))
        for minos_str, run_ids in sorted_results:
            print "cnt={}, mids={}, minos_str: {}".format(len(run_ids), run_ids, minos_str)
        print "\n{} - Done running ({} sec)".format(dt.datetime.now(), timer() - start)


if __name__ == '__main__':
    main()
