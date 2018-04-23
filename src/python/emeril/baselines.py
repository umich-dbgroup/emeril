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
from . import *
from .utils import get_target_count_bounds
import psycopg2


###############################################################################
############################ Tiresias baseline ################################

def get_tiresias_baseline_data(tuple_ids, bin_ids, pred_ids, bin_tuple_ids,
                               tuple_preds, target_counts, target_bounds):
    # 5. output mathprog
    out = "set tuple_ids := {};\n".format(' '.join(map(str, tuple_ids)))
    out += "set bin_ids := {};\n".format(' '.join(map(str, bin_ids)))
    out += "set pred_ids := {};\n".format(' '.join(map(str, pred_ids)))
    for bin_id in bin_ids:
        cur_ids = bin_tuple_ids[bin_id]
        out += "set bin_tuple_ids[{}] := {};\n".format(bin_id, ', '.join(map(str, cur_ids)))
    out += "param tuple_preds: {} :=\n".format(' '.join(map(str, pred_ids)))
    for tid in tuple_ids:
        cur_row_vals = np.zeros(len(pred_ids), dtype=int)
        for pid in tuple_preds[tid]:
            cur_row_vals[pred_ids.index(pid)] = 1
        out += "{}  {}\n".format(tid, ' '.join(map(str, cur_row_vals)))
    out += ";\n"
    out += "param num_bins := {};\n".format(len(bin_ids))
    out += "param num_preds := {};\n".format(len(pred_ids))
    out += "param target_counts :=\n"
    for bin_id, target_count in enumerate(target_counts):
        out += "  {} {}\n".format(bin_id, target_count)
    out += ";\n"
    out += "param target_bounds: lower upper :=\n"
    for bin_id, (lower, upper) in enumerate(target_bounds):
        out += "  {} {} {}\n".format(bin_id, lower, upper)
    out += ";\n"
    out += "end;\n"
    return out


def get_tiresias_baseline_data_from_db(db, table, pk_field, target_field, target_signal,
                                       adjusted_bin_edges, pred_table, pred_pk, pred_map_table,
                                       slack_percent):
    con = psycopg2.connect("dbname='{}' user='{}' host='{}' password='{}'"
                           .format(db, 'emeril', 'localhost', 'emeril'))
    cur = con.cursor()

    # 1. get target signal & bin_ids
    target_counts = map(int, target_signal.split(','))
    target_bounds = get_target_count_bounds(target_counts, slack_percent)
    bin_ids = range(len(target_counts))

    # 2. get tuples from table, putting into bins
    adjusted_bin_edges = map(float, adjusted_bin_edges.split(','))
    sql = "SELECT {}, \"{}\" FROM {}".format(pk_field, target_field, table)
    cur.execute(sql)
    rows = cur.fetchall()
    tuple_ids = []
    vals = []
    bin_tuple_ids = defaultdict(list)
    for pk, val in rows:
        tuple_ids.append(pk)
        vals.append(val)
        found_bin = False
        for bin_id, edge in enumerate(adjusted_bin_edges[1:]):
            # print("val = {}, edge = {}, type(val)={}, type(edge)={}".format(val, edge, type(val), type(edge)))
            # exit()
            if float(val) < edge:
                found_bin = True
                bin_tuple_ids[bin_id].append(pk)
                break
        if not found_bin:
            bin_tuple_ids[bin_id].append(pk)  # using last bin

    # 3. getting predicates
    sql = "SELECT {} FROM {}".format(pred_pk, pred_table)
    cur.execute(sql)
    pred_ids = [x[0] for x in cur.fetchall()]

    # 4. getting predicate mappings
    tuple_preds = defaultdict(list)
    sql = "SELECT {}, {} FROM {}".format(pred_pk, pk_field, pred_map_table)
    cur.execute(sql)
    rows = cur.fetchall()
    for pid, tid in rows:
        tuple_preds[tid].append(pid)

    # 5. return mathprog data
    return get_tiresias_baseline_data(tuple_ids, bin_ids, pred_ids, bin_tuple_ids,
                                      tuple_preds, target_counts, target_bounds)







###############################################################################
############################# ConQuER baseline ################################

def run_conquer_solver_v1(df, preds, target_counts, target_fld, adjusted_bin_edges,
                          target_answer_pids, print_details=True):
    """
    Runs conquer baseline on dataset
    """
    meta = {}

    # 1. do conquer solving method
    raise Exception("")


def get_conquer_query(df, orig_query, product_name):
    """
    Returns conquer's query for given dataframe + where clauses + whynot tuples

    Assumptions:
    (1) only one whynot tuple (product_name = Strawberry Lemonade)
        Others: Granola, Tomato Ketchup
    (2) assuming "<=" is predicate operator
    (3) assume 3x |input predicates| allowed in refined query (ConQueR did in experiments)

    ToDO: repeat 1-4 below for each why-not tuple; query value is max across tuples
    """
    # 1. getting max values for where fields
    orig_bool_query = ' AND '.join(['{} <= {}'.format(f, v) for f, v in orig_query])
    orig_results = df.query(orig_bool_query)
    orig_maxes = {}
    for (where_fld, where_val) in orig_query:
        orig_maxes[where_fld] = orig_results[where_fld].max()

    # 2. get M_1 tuples
    m1 = df[df['product_name'] == product_name]

    # 3. filter "skyline tuples" from M1 (all attribs <=, and at least one <)
    # <skipping; just an optimization?>

    # 4. for each tuple in m1, create refined query from max{v_1_max and t_i.A_t}
    queries = []
    for index, row in m1.iterrows():
        predicates = []
        for (where_fld, where_val) in orig_query:
            refined_val = max(orig_maxes[where_fld], row[where_fld])
            predicates.append((where_fld, refined_val))
        queries.append(predicates)

    # 5. getting unused attribs (for use in improving precision)
    unused_cols = list(numeric_fields)
    for (where_fld, where_val) in orig_query:
        unused_cols.remove(where_fld)

    # 6. getting unused attrib maxes
    unused_col_maxes = {}
    orig_plus_m = df.query('({}) or product_name == "{}"'.format(orig_bool_query, product_name))
    bad_unused_cols = []
    for col in unused_cols:
        unused_col_max = orig_plus_m[col].max()
        if unused_col_max and not np.isnan(unused_col_max):
            unused_col_maxes[col] = unused_col_max
        else:
            bad_unused_cols.append(col)
    for bad in bad_unused_cols:
        unused_cols.remove(bad)

    # 7. get scores for each query
    scores = []
    new_queries = []
    for predicates in queries:
        # 7a. getting imprecision
        cur = df.query(' and '.join(['{} <= {}'.format(f, v) for f, v in predicates]))
        imprecision = len(cur[~cur.isin(orig_results) & ~cur.isin(m1)].dropna(how='all'))

        # 7b. finding effectiveness of unused columns
        new_imprecisions = []
        for col in unused_cols:
            new_imprecisions.append(len(cur[cur[col] < unused_col_maxes[col]]))

        # 7c. using top X most efficient unused column predicates
        top_indexes = np.argsort(new_imprecisions)[::-1][0:3*len(orig_query)]
        new_where = [(unused_cols[i], unused_col_maxes[unused_cols[i]]) for i in top_indexes]
        new_where += predicates
        new_cur = df.query(' and '.join(['{} <= {}'.format(f, v) for f, v in new_where]))
        new_imprecision = len(new_cur[~new_cur.isin(orig_results) & ~new_cur.isin(m1)].dropna(how='all'))
        scores.append(new_imprecision)
        new_queries.append(new_where)

    # 8. return best query and imprecision score
    min_score_index = np.argmin(scores)
    return new_queries[min_score_index], scores[min_score_index]








###############################################################################
############################# n-choose-k baseline #############################

def run_n_choose_k_solver_v1(df, preds, target_counts, target_fld, adjusted_bin_edges,
                             target_answer_pids, slack_percent=0.2,
                             print_details=True, max_preds=2, sorting_random_seed=RANDOM_SEED,
                             timeout=None):
    """
    Runs n-choose-k baseline on dataset
    """
    meta = {}

    # determine total combos
    total_combos = 0
    for i in range(1, max_preds+1):
        total_combos += math.factorial(len(preds)) / (math.factorial(i) * math.factorial(len(preds) - i))
    if print_details:
        print "total n-choose-k options: {}".format(total_combos)

    # 1. shuffling pred ids
    np_random = np.random.RandomState(sorting_random_seed)
    rand_pred_ids = range(0, len(preds))
    np_random.shuffle(rand_pred_ids)

    # 2. get pred pairs
    if timeout is not None:
        print "{} - Starting nchoosek search; timeout set to {} seconds".format(dt.datetime.now(), timeout)
    else:
        print "{} - Starting nchoosek search; no timeout set".format(dt.datetime.now())
    start = timer()
    results = []
    for i in range(1, max_preds+1):
        if print_details:
            num_combos = math.factorial(len(preds)) / (math.factorial(i) * math.factorial(len(preds) - i))
            print "starting n-choose-{} from {} preds, num_combos={} (total={})".format(i, len(preds), num_combos, total_combos)
        for cur_rand_index_ids in itertools.combinations(range(len(preds)), i):
            query = Query(inverted_predicates=False)
            cur_pred_ids = [rand_pred_ids[x] for x in cur_rand_index_ids]
            for pid in cur_pred_ids:
                query.add_predicate(*preds[pid])
            answer_df = df.query(query.get_pandas_query())
            answer_distrib = np.histogram(answer_df[target_fld], adjusted_bin_edges)[0]
            signal_dist = sum([abs(target_counts[i] - answer_distrib[i]) for i in xrange(len(target_counts))])
            #corr = pearsonr(target_counts, answer_distrib)[0]
            results.append((signal_dist, cur_pred_ids))
            if timeout is not None and timer() - start > timeout:
                break
        if timeout is not None and timer() - start > timeout:
            print "Terminating n-choose-{} after {} results (timer done at {} sec)"\
                .format(max_preds, len(results), timer() - start)
            break

    # 3. sorting to get top result
    results.sort(key=lambda x: x[0], reverse=False)
    top_result = results[0]

    mip_answer_ids = top_result[1]
    answer_preds = [pred for pid, pred in enumerate(preds) if pid in mip_answer_ids]
    meta['final_answer_pids'] = mip_answer_ids
    meta['solution_sp'] = slack_percent
    meta['mip_runtimes'] = [timer() - start, ]
    meta['mip_outputs'] = ["not applicable"]
    meta['mip_statuses'] = ["not applicable"]
    meta['slack_percents'] = [slack_percent]
    meta['minos_opt_strs'] = [None]

    query = Query(inverted_predicates=False)
    for pred in answer_preds:
        query.add_predicate(*pred)
    qs = query.get_pandas_query()
    if not qs:
        print "ERROR: no qs"
        print mip_output
        raise Exception("No qs.. wtf")
    answer_df = get_df_from_many_preds(df, answer_preds)
    answer_distrib = np.histogram(answer_df[target_fld], adjusted_bin_edges)[0]

    meta['solution_found'] = True
    meta['is_valid_solution'] = True
    target_bounds = get_target_count_bounds(target_counts, slack_percent)
    for bin_id, (lower, upper) in enumerate(target_bounds):
        if answer_distrib[bin_id] < lower or answer_distrib[bin_id] > upper:
            meta['is_valid_solution'] = False
            break
    meta['exact_solution'] = (tuple(meta['final_answer_pids']) == tuple(target_answer_pids))
    meta['semi_exact'] = meta['is_valid_solution'] and all(x in meta['final_answer_pids'] for x in target_answer_pids)

    # 3a. getting answer query and resulting histogram
    if print_details:
        print "query: {}".format(qs)
        print "len answer: {}".format(len(answer_df))
        print "Distrib of fld='{}' of full dataset: {}".format(target_fld, np.histogram(df[target_fld], adjusted_bin_edges)[0])
        print "Target bounds: {}".format(target_bounds)
        print "Distrib of fld='{}' w/ MIP solution's query applied: {}".format(target_fld, answer_distrib)
        print "is_valid_solution = {}, exact? {}".format(yes_no(meta['is_valid_solution']), yes_no(meta['exact_solution']))

    # 4. returning meta
    return meta






###############################################################################
############################## Greedy baseline ################################

def run_greedy_solver_v1(df, preds, target_counts, target_fld, adjusted_bin_edges,
                         target_answer_pids, slack_percent=0.2, print_details=True,
                         max_preds=2):
    """
    Runs greedy baseline on dataset
    """
    meta = {}

    # 1. get pred pairs
    start = timer()
    results = []
    best_of_k = []  # [0] has best of k=1; [1] has best of k=2 w/ one being best_of_k[0]
    for k in xrange(1, max_preds+1):
        cur_k_results = []
        for pid, pred in enumerate(preds):
            # 1a. skipping previously used bests
            if pid in best_of_k:
                continue

            # 1b. building query of previously used bests and current pred
            query = Query(inverted_predicates=False)
            query.add_predicate(*pred)
            for best_pid in best_of_k:
                query.add_predicate(*preds[best_pid])

            # 1c. getting score
            answer_df = df.query(query.get_pandas_query())
            answer_distrib = np.histogram(answer_df[target_fld], adjusted_bin_edges)[0]
            signal_dist = sum([abs(target_counts[i] - answer_distrib[i]) for i in xrange(len(target_counts))])
            #corr = pearsonr(target_counts, answer_distrib)[0]
            cur_k_results.append((signal_dist, pid))

        # 1d. determining best pred addition
        cur_k_results.sort(key=lambda x: x[0], reverse=False)
        if print_details:
            print("greedy search k={}, best={}".format(k, cur_k_results[0]))
        best_of_k.append(cur_k_results[0][1])

        # 1e. recording all results
        for signal_dist, pid in cur_k_results:
            results.append((signal_dist, best_of_k[0:-1] + [pid]))

    # 2. sorting to get top result
    results.sort(key=lambda x: x[0], reverse=False)
    top_result = results[0]
    if print_details:
        print("Done finding best answer: {} sec".format(timer() - start))
        print("Best of k: {}".format(best_of_k))
        print("Top result: {}".format(top_result))


    # 3. getting results from pred ids
    mip_answer_ids = top_result[1]
    answer_preds = [pred for pid, pred in enumerate(preds) if pid in mip_answer_ids]
    meta['final_answer_pids'] = mip_answer_ids
    meta['solution_sp'] = slack_percent
    meta['mip_runtimes'] = [timer() - start, ]
    meta['mip_outputs'] = ["not applicable"]
    meta['mip_statuses'] = ["not applicable"]
    meta['slack_percents'] = [slack_percent]
    meta['minos_opt_strs'] = [None]

    query = Query(inverted_predicates=False)
    for pred in answer_preds:
        query.add_predicate(*pred)
    qs = query.get_pandas_query()
    if not qs:
        print "ERROR: no qs"
        print mip_output
        raise Exception("No qs.. wtf")
    answer_df = get_df_from_many_preds(df, answer_preds)
    answer_distrib = np.histogram(answer_df[target_fld], adjusted_bin_edges)[0]

    meta['solution_found'] = True
    meta['is_valid_solution'] = True
    target_bounds = get_target_count_bounds(target_counts, slack_percent)
    for bin_id, (lower, upper) in enumerate(target_bounds):
        if answer_distrib[bin_id] < lower or answer_distrib[bin_id] > upper:
            meta['is_valid_solution'] = False
            break
    meta['exact_solution'] = (tuple(meta['final_answer_pids']) == tuple(target_answer_pids))
    meta['semi_exact'] = meta['is_valid_solution'] and all(x in meta['final_answer_pids'] for x in target_answer_pids)

    # 3a. getting answer query and resulting histogram
    if print_details:
        print "query: {}".format(qs)
        print "len answer: {}".format(len(answer_df))
        print "Distrib of fld='{}' of full dataset: {}".format(target_fld, np.histogram(df[target_fld], adjusted_bin_edges)[0])
        print "Target bounds: {}".format(target_bounds)
        print "Distrib of fld='{}' w/ MIP solution's query applied: {}".format(target_fld, answer_distrib)
        print "is_valid_solution = {}, exact? {}".format(yes_no(meta['is_valid_solution']), yes_no(meta['exact_solution']))

    # 4. returning meta
    return meta





###############################################################################
########################### Freq mining baseline ##############################

def run_freq_mining_solver_v1(df, preds, target_counts, target_fld,
                              adjusted_bin_edges, target_answer_pids,
                              print_details=True):
    raise Exception("Freq mining used to find dependent preds; still requires mip solver. "
                    "Maybe will be useful as comparison vs. random or our sorting?")


def create_freq_mining_pred_transactions(db, pk, table, pred_table, pred_pk,
                                         pred_field_name, output_file):
    con = psycopg2.connect("dbname='{}' user='{}' host='{}' password='{}'"
                           .format(args.db, 'emeril', 'localhost', 'emeril'))
    cur = con.cursor()

    # 1. load predicates, determine support for each
    tuple_preds = defaultdict(list)
    sql = "SELECT {}, {} FROM {} ORDER by {}"\
        .format(args.pred_pk, args.pred_field_name, args.pred_table, args.pred_pk)
    cur.execute(sql)
    for pid, pred in cur.fetchall():
        safe_pred = get_safe_pred(pred)
        sql = "SELECT {} FROM {} WHERE {}".format(args.pk, args.table, safe_pred)
        cur.execute(sql)
        for pk in cur.fetchall():
            tuple_preds[pk].append(pid)

    # 2. convert to list
    tuple_preds_list = []
    for pk in sorted(tuple_preds.keys()):
        tuple_preds_list.append(tuple_preds[pk])

    # 3. save to pickle
    # To open: with open(filename, 'rb') as f: tuple_preds_list = pickle.load(f)
    with open(args.output_file, 'w') as f:
        pickle.dump(tuple_preds_list, f, -1)


def find_freq_mining_pred_deps(infile, outfile, mode='fim.fpgrowth',
                               support=10, confidence=80,
                               zmin=2, zmax=2, sample_to_print=5):
    with open(infile) as f:
        transactions = pickle.load(f)

    if mode == 'fim.fpgrowth':
        import fim
        patterns = fim.fpgrowth(transactions, zmin=zmin, zmax=zmax,
                                supp=support, conf=confidence)
        print "## Sample of rules ({} total): ##".format(len(patterns))
        print patterns[0:sample_to_print]

    elif mode == 'fim.carpenter':
        import fim
        patterns = fim.carpenter(transactions, zmin=2, zmax=2)
        print "## Sample of rules ({} total): ##".format(len(patterns))
        print patterns[0:sample_to_print]

    with open(outfile, 'w') as f:
        pickle.dump(patterns, f, -1)
