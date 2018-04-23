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
from syndata import get_data_cache_key

###############################################################################
######################### Oldish Evaluation ###################################

def get_signal_sim(df, db_meta, constraints, cand_query):
    ## 1. getting target signal and field ##
    # TODO(Dolan): add support for more than one target
    if len(constraints['want_change_signals']) > 1:
        raise Exception("UNSUPPORTED: only supports one want_change_signals")
    target_field, target_signal = constraints['want_change_signals'][0]
    result_data = df.query(cand_query.get_pandas_query())[target_field]
    bin_edges = db_meta.get_field_bins(target_field)
    result_signal, _ = np.histogram(result_data, bins=bin_edges)

    signal_sim = get_pearson_sim(target_signal, result_signal)
    signal_sim = 0.0 if signal_sim < 0.0 else signal_sim
    # # signal_sim = get_kl_divergence(target_signal, result_signal, fake_normalize_sim=True)
    # target_percents = [x / float(sum(target_signal)) for x in target_signal]
    # result_percents = [x / float(sum(result_signal)) for x in result_signal]
    # signal_sim = get_mape_sim(target_percents, result_percents)
    return signal_sim


def get_code_sim(df, db_meta, constraints, orig_query, cand_query):
    # TODO: implement
    return (1 - (len(cand_query.predicates) / float(MAX_PREDS))) + REALLY_SMALL_NUMBER


def emeril_algo4_score_queries(db_meta, df, constraints, candidate_queries, orig_query):
    """
    scores queries and returns skyline for "distrib" and "code" similarities
    """
    print("\n====== Starting query scoring =========")

    ## 1. score candidate queries (for all beta values) ##
    start = timer()
    query_scores = defaultdict(list)
    for query_id, cand_query in enumerate(candidate_queries):
        # 1a. getting signal and code fscores
        signal_sim = get_signal_sim(df, db_meta, constraints, cand_query)
        code_sim = get_code_sim(df, db_meta, constraints, orig_query, cand_query)
        for beta in QUERY_SCORE_BETAS:
            if beta == 'allsig':
                fscore = signal_sim
            elif beta == 'allcode':
                fscore = code_sim
            else:
                if code_sim == 0:
                    raise Exception("fld = {}".format(fld))
                fscore = get_fscore(signal_sim, code_sim, beta=beta)
            query_scores[beta].append((fscore, signal_sim, code_sim, cand_query))

        # 1b. status update
        if query_id % 200 == 0:
            print("Evaluating query {} of {} (at {:.2f} seconds)..."
                  .format(query_id, len(candidate_queries), timer() - start))

    ## 2. sorting queries for each beta ##
    for beta in QUERY_SCORE_BETAS:
        query_scores[beta].sort(reverse=True, key=lambda x: x[0])

    ## 3. getting target signal and field ##
    # TODO(Dolan): add support for more than one target
    if len(constraints['want_change_signals']) > 1:
        raise Exception("UNSUPPORTED: only supports one want_change_signals")
    target_field, target_signal = constraints['want_change_signals'][0]
    field_type = db_meta.get_field_type(target_field)
    bin_edges = db_meta.get_field_bins(target_field)
    orig_counts, _ = np.histogram(df[target_field], bins=bin_edges)
    orig_percents = [x / float(sum(orig_counts)) for x in orig_counts]

    ## 3. printing top results for each beta ##
    num_to_output = 3
    print("\nPrinting top {} results for each beta...".format(num_to_output))
    for beta in ('allsig', ): #QUERY_SCORE_BETAS:
        print("\n-------------------- beta={} -----------------------".format(beta))
        for pos, (fscore, signal_sim, code_sim, cand_query) in enumerate(query_scores[beta][0:num_to_output]):
            result_data = df.query(cand_query.get_pandas_query())[target_field]
            result_signal, _ = np.histogram(result_data, bins=bin_edges)
            result_percents = [x / float(sum(result_signal)) for x in result_signal]

            orig_corr = get_pearson_sim(result_signal, orig_counts)
            print("signal_sim={:.2f}, orig_corr={:.2f}, code_sim={:.2f}, fscore={:.2f}\n{}"
                  .format(signal_sim, orig_corr, code_sim, fscore, cand_query.get_pandas_query()))
            print("orig = {}".format(", ".join(['{:.2f}'.format(x) for x in orig_percents])))
            print("result = {}".format(", ".join(['{:.2f}'.format(x) for x in result_percents])))
            print("target = {}".format(", ".join(['{:.2f}'.format(x) for x in target_signal])))
            if field_type == 'categorical':
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(15, 5))
                plot1 = sns.barplot(x=bin_edges[:-1], y=orig_percents, ax=ax1)
                plot1.axes.set_ylim(0.0, 0.6)
                plot2 = sns.barplot(x=bin_edges[:-1], y=result_percents, ax=ax2)
                plot2.axes.set_ylim(0.0, 0.6)
                plot3 = sns.barplot(x=bin_edges[:-1], y=target_signal, ax=ax3)
                plot3.axes.set_ylim(0.0, 0.6)
                plt.show()
            else:
                raise Exception("Not implemented")
            print('')





###############################################################################
########################## Extract corr results ###############################

def get_corr_exp_results(system, num_rows, num_columns, num_runs, slack_percent,
                         max_preds=2, corrs=np.arange(0.0, 1.1, 0.1),
                         pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                         syn_data_mode="v1", num_preds=None,
                         meta_cache_dir=DATA_GEN_EXP_META_DIR):
    stats = {}
    stats['minos_distrib'] = defaultdict(lambda: defaultdict(int))
    stats['missed_valid_answers'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['corr_accuracy_stats'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['corr_runtime_stats'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['final_status_result'] = defaultdict(int)
    stats['answer_dep_ranks'] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    stats['corr_statuses'] = defaultdict(lambda: defaultdict(list))
    stats['corr_hybrid_solver'] = defaultdict(lambda: defaultdict(list))
    for corr in corrs:
        for pred_dep_percent in pred_dep_percents:
            for i in xrange(num_runs):
                if system in ("tiresias", "conquer"):
                    cur_sys = system
                elif system == "n_choose_k":
                    cur_sys = system[0:-1] + str(max_preds)
                elif system == "greedy":
                    cur_sys = system + str(max_preds)
                elif system == "emerilRandom":
                    cur_sys = "emDep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilIndep":
                    cur_sys = "emIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilHybrid":
                    cur_sys = "emHybrid{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresRand":
                    cur_sys = "emThresRand{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresRandIndep":
                    cur_sys = "emThresRandIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresIndep":
                    cur_sys = "emThresIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilTriHybrid":
                    cur_sys = "emTriHybrid{}".format(int(pred_dep_percent * 100))
                elif system == "emerilTiHybrid":
                    cur_sys = "emTrHybrid{}".format(int(pred_dep_percent * 100))
                random_seed = RANDOM_SEED + i
                sorting_random_seed = RANDOM_SEED + i
                data_cache_key = get_data_cache_key(syn_data_mode, num_rows, num_columns, num_preds, corr, random_seed)
                meta_cache_key = data_cache_key + '-sys.{}-pdp{}-sort{}-sp{}'\
                                    .format(cur_sys, pred_dep_percent, sorting_random_seed, slack_percent)
                meta_cache_file = os.path.join(meta_cache_dir, syn_data_mode, meta_cache_key + ".cache")
                with open(meta_cache_file) as f:
                    meta = pickle.load(f)

                # 1. determine which minos options work best
                for j, minos_opt_str in enumerate(meta['minos_opt_strs']):
                    status = meta['mip_statuses'][j]
                    stats['minos_distrib'][minos_opt_str][status] += 1

                # 2. compare answer dep rank vs. results
                if 'answer_pids_dep_scores_rank' in meta:
                    answer_included = meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
                    missed_found = not meta['solution_found'] and answer_included
                    missed_valid = not meta['is_valid_solution'] and answer_included
                    missed_semiexact = not meta['semi_exact'] and answer_included
                    missed_exact = not meta['exact_solution'] and answer_included
                    stats['missed_valid_answers'][corr][pred_dep_percent]['found'] += 1 if missed_found else 0
                    stats['missed_valid_answers'][corr][pred_dep_percent]['valid'] += 1 if missed_valid else 0
                    stats['missed_valid_answers'][corr][pred_dep_percent]['semiexact'] += 1 if missed_semiexact else 0
                    stats['missed_valid_answers'][corr][pred_dep_percent]['exact'] += 1 if missed_exact else 0
                    stats['missed_valid_answers'][corr][pred_dep_percent]['answer_included'] += 1 if answer_included else 0

                # 3. getting accuracy stats
                stats['corr_accuracy_stats'][corr][pred_dep_percent]['found'] += 1 if meta['solution_found'] else 0
                stats['corr_accuracy_stats'][corr][pred_dep_percent]['valid'] += 1 if meta['is_valid_solution'] else 0
                stats['corr_accuracy_stats'][corr][pred_dep_percent]['semiexact'] += 1 if meta['semi_exact'] else 0
                stats['corr_accuracy_stats'][corr][pred_dep_percent]['exact'] += 1 if meta['exact_solution'] else 0

                # 4. extract runtime for different experiments
                stats['corr_runtime_stats'][corr][pred_dep_percent]['total'] += sum(meta['mip_runtimes'])
                stats['corr_runtime_stats'][corr][pred_dep_percent]['last'] += meta['mip_runtimes'][-1]

                # 5. final status/result
                status = meta['mip_statuses'][-1]
                is_valid = meta['is_valid_solution']
                if 'answer_pids_dep_scores_rank' in meta:
                    answer_included = meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
                else:
                    answer_included = False
                stats['final_status_result'][(status, is_valid, answer_included)] += 1

                # 6. answer dep ranks
                if 'answer_pids_dep_scores_rank' in meta:
                    stats['answer_dep_ranks'][corr][pred_dep_percent]['rank'].append(meta['answer_pids_dep_scores_rank'])
                    stats['answer_dep_ranks'][corr][pred_dep_percent]['used'].append(meta['num_dep_scores_used'])
                    stats['answer_dep_ranks'][corr][pred_dep_percent]['avail'].append(meta['num_dep_scores_avail'])

                # 7. statuses
                if status == 'infeasible':
                    status = 'infeasible_' + ('has_ans' if answer_included else 'no_ans')
                stats['corr_statuses'][corr][pred_dep_percent].append(status)

                # 8. noting hybrid solver, if applicable
                stats['corr_hybrid_solver'][corr][pred_dep_percent].append(meta.get('hybrid_solver', 'n/a'))

            # creating averages for accuracy & runtime stats
            stats['corr_accuracy_stats'][corr][pred_dep_percent]['found'] /= float(num_runs)
            stats['corr_accuracy_stats'][corr][pred_dep_percent]['valid'] /= float(num_runs)
            stats['corr_accuracy_stats'][corr][pred_dep_percent]['semiexact'] /= float(num_runs)
            stats['corr_accuracy_stats'][corr][pred_dep_percent]['exact'] /= float(num_runs)
            stats['corr_runtime_stats'][corr][pred_dep_percent]['total'] /= float(num_runs)
            stats['corr_runtime_stats'][corr][pred_dep_percent]['last'] /= float(num_runs)

    return stats


def print_corr_accuracy_results(corr_accuracy_stats, corrs=np.arange(0.0, 1.1, 0.1),
                                pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                                keys=["found", 'valid', 'semiexact', 'exact']):
    for corr in corrs:
        row = []
        for key in keys:
            row += [corr_accuracy_stats[corr][pdp][key] for pdp in pred_dep_percents]
        print "\t".join(map(str, row))


def print_corr_runtime_results(corr_runtime_stats, corrs=np.arange(0.0, 1.1, 0.1),
                               pred_dep_percents=np.arange(0.0, 1.1, 0.1), keys=["total", "last"]):
    for corr in corrs:
        row = []
        for key in keys:
            row += [corr_runtime_stats[corr][pdp][key] for pdp in pred_dep_percents]
        print "\t".join(map(str, row))





###############################################################################
########################## Extract row results ################################

def get_row_exp_results(system, corr, num_columns, num_runs, slack_percent,
                         max_preds=2, rows=range(100000, 1000001, 100000),
                         pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                         syn_data_mode="v1", num_preds=None):
    stats = {}
    stats['minos_distrib'] = defaultdict(lambda: defaultdict(int))
    stats['missed_valid_answers'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['row_accuracy_stats'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['row_runtime_stats'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['final_status_result'] = defaultdict(int)
    stats['answer_dep_ranks'] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    stats['row_statuses'] = defaultdict(lambda: defaultdict(list))
    for num_rows in rows:
        for pred_dep_percent in pred_dep_percents:
            for i in xrange(num_runs):
                if system in ("tiresias", "conquer"):
                    cur_sys = system
                elif system == "n_choose_k":
                    cur_sys = system[0:-1] + str(max_preds)
                elif system == "greedy":
                    cur_sys = system + str(max_preds)
                elif system == "emerilRandom":
                    cur_sys = "emDep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilIndep":
                    cur_sys = "emIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilHybrid":
                    cur_sys = "emHybrid{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresRand":
                    cur_sys = "emThresRand{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresRandIndep":
                    cur_sys = "emThresRandIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresIndep":
                    cur_sys = "emThresIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilTriHybrid":
                    cur_sys = "emTriHybrid{}".format(int(pred_dep_percent * 100))
                elif system == "emerilTiHybrid":
                    cur_sys = "emTrHybrid{}".format(int(pred_dep_percent * 100))
                random_seed = RANDOM_SEED + i
                sorting_random_seed = RANDOM_SEED + i
                data_cache_key = get_data_cache_key(syn_data_mode, num_rows, num_columns, num_preds, corr, random_seed)
                meta_cache_key = data_cache_key + '-sys.{}-pdp{}-sort{}-sp{}'\
                                    .format(cur_sys, pred_dep_percent, sorting_random_seed, slack_percent)
                meta_cache_file = os.path.join(DATA_GEN_EXP_META_DIR, syn_data_mode, meta_cache_key + ".cache")
                with open(meta_cache_file) as f:
                    meta = pickle.load(f)

                # 1. determine which minos options work best
                for j, minos_opt_str in enumerate(meta['minos_opt_strs']):
                    status = meta['mip_statuses'][j]
                    stats['minos_distrib'][minos_opt_str][status] += 1

                # 2. compare answer dep rank vs. results
                if 'answer_pids_dep_scores_rank' in meta:
                    answer_included = meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
                    missed_found = not meta['solution_found'] and answer_included
                    missed_valid = not meta['is_valid_solution'] and answer_included
                    missed_semiexact = not meta['semi_exact'] and answer_included
                    missed_exact = not meta['exact_solution'] and answer_included
                    stats['missed_valid_answers'][num_rows][pred_dep_percent]['found'] += 1 if missed_found else 0
                    stats['missed_valid_answers'][num_rows][pred_dep_percent]['valid'] += 1 if missed_valid else 0
                    stats['missed_valid_answers'][num_rows][pred_dep_percent]['semiexact'] += 1 if missed_semiexact else 0
                    stats['missed_valid_answers'][num_rows][pred_dep_percent]['exact'] += 1 if missed_exact else 0
                    stats['missed_valid_answers'][num_rows][pred_dep_percent]['answer_included'] += 1 if answer_included else 0

                # 3. getting accuracy stats
                stats['row_accuracy_stats'][num_rows][pred_dep_percent]['found'] += 1 if meta['solution_found'] else 0
                stats['row_accuracy_stats'][num_rows][pred_dep_percent]['valid'] += 1 if meta['is_valid_solution'] else 0
                stats['row_accuracy_stats'][num_rows][pred_dep_percent]['semiexact'] += 1 if meta['semi_exact'] else 0
                stats['row_accuracy_stats'][num_rows][pred_dep_percent]['exact'] += 1 if meta['exact_solution'] else 0

                # 4. extract runtime for different experiments
                stats['row_runtime_stats'][num_rows][pred_dep_percent]['total'] += sum(meta['mip_runtimes'])
                stats['row_runtime_stats'][num_rows][pred_dep_percent]['last'] += meta['mip_runtimes'][-1]

                # 5. final status/result
                status = meta['mip_statuses'][-1]
                is_valid = meta['is_valid_solution']
                if 'answer_pids_dep_scores_rank' in meta:
                    answer_included = meta['num_dep_scores_used'] and meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
                else:
                    answer_included = False
                stats['final_status_result'][(status, is_valid, answer_included)] += 1

                # 6. answer dep ranks
                if 'answer_pids_dep_scores_rank' in meta:
                    stats['answer_dep_ranks'][num_rows][pred_dep_percent]['rank'].append(meta['answer_pids_dep_scores_rank'])
                    stats['answer_dep_ranks'][num_rows][pred_dep_percent]['used'].append(meta['num_dep_scores_used'])
                    stats['answer_dep_ranks'][num_rows][pred_dep_percent]['avail'].append(meta['num_dep_scores_avail'])

                # 7. statuses
                if status == 'infeasible':
                    status = 'infeasible_' + ('has_ans' if answer_included else 'no_ans')
                stats['row_statuses'][num_rows][pred_dep_percent].append(status)

            # creating averages for accuracy & runtime stats
            stats['row_accuracy_stats'][num_rows][pred_dep_percent]['found'] /= float(num_runs)
            stats['row_accuracy_stats'][num_rows][pred_dep_percent]['valid'] /= float(num_runs)
            stats['row_accuracy_stats'][num_rows][pred_dep_percent]['semiexact'] /= float(num_runs)
            stats['row_accuracy_stats'][num_rows][pred_dep_percent]['exact'] /= float(num_runs)
            stats['row_runtime_stats'][num_rows][pred_dep_percent]['total'] /= float(num_runs)
            stats['row_runtime_stats'][num_rows][pred_dep_percent]['last'] /= float(num_runs)

    return stats


def print_row_accuracy_results(row_accuracy_stats, rows=range(100000, 1000001, 100000),
                                pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                                keys=["found", 'valid', 'semiexact', 'exact']):
    for num_rows in rows:
        row = []
        for key in keys:
            row += [row_accuracy_stats[num_rows][pdp][key] for pdp in pred_dep_percents]
        print "\t".join(map(str, row))


def print_row_runtime_results(row_runtime_stats, rows=range(100000, 1000001, 100000),
                               pred_dep_percents=np.arange(0.0, 1.1, 0.1), keys=["total", "last"]):
    for num_rows in rows:
        row = []
        for key in keys:
            row += [row_runtime_stats[num_rows][pdp][key] for pdp in pred_dep_percents]
        print "\t".join(map(str, row))






###############################################################################
########################## Extract pred results ###############################

def get_pred_exp_results(system, num_rows, num_columns, num_runs, slack_percent,
                         max_preds=2, preds=range(1000, 10001, 1000),
                         pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                         meta_key_timeout=None, syn_data_mode="v2",
                         use_code_sim=False, corr=None):
    stats = {}
    stats['minos_distrib'] = defaultdict(lambda: defaultdict(int))
    stats['missed_valid_answers'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['pred_accuracy_stats'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['pred_runtime_stats'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['final_status_result'] = defaultdict(int)
    stats['answer_dep_ranks'] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    stats['pred_statuses'] = defaultdict(lambda: defaultdict(list))
    for num_preds in preds:
        for pred_dep_percent in pred_dep_percents:
            for i in xrange(num_runs):
                if system in ("tiresias", "conquer"):
                    cur_sys = system
                elif system == "n_choose_k":
                    cur_sys = system[0:-1] + str(max_preds)
                elif system == "greedy":
                    cur_sys = system + str(max_preds)
                elif system == "emerilRandom":
                    cur_sys = "emDep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilIndep":
                    cur_sys = "emIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilHybrid":
                    cur_sys = "emHybrid{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresRand":
                    cur_sys = "emThresRand{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresRandIndep":
                    cur_sys = "emThresRandIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilThresIndep":
                    cur_sys = "emThresIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilTriHybrid":
                    cur_sys = "emTriHybrid{}".format(int(pred_dep_percent * 100))
                elif system == "emerilTiHybrid":
                    cur_sys = "emTrHybrid{}".format(int(pred_dep_percent * 100))
                random_seed = RANDOM_SEED + i
                sorting_random_seed = RANDOM_SEED + i
                data_cache_key = get_data_cache_key(syn_data_mode, num_rows, num_columns, num_preds, corr, random_seed)
                meta_cache_key = data_cache_key + '-sys.{}-pdp{}-sort{}-sp{}'\
                                    .format(cur_sys, pred_dep_percent, sorting_random_seed, slack_percent)
                if meta_key_timeout:
                    meta_cache_key += '-time{}'.format(meta_key_timeout)
                if use_code_sim:
                    meta_cache_key += '-codesim'
                meta_cache_file = os.path.join(DATA_GEN_EXP_META_DIR, syn_data_mode, meta_cache_key + ".cache")
                with open(meta_cache_file) as f:
                    meta = pickle.load(f)

                # 1. determine which minos options work best
                for j, minos_opt_str in enumerate(meta['minos_opt_strs']):
                    status = meta['mip_statuses'][j]
                    stats['minos_distrib'][minos_opt_str][status] += 1

                # 2. compare answer dep rank vs. results
                if 'answer_pids_dep_scores_rank' in meta:
                    answer_included = meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
                    missed_found = not meta['solution_found'] and answer_included
                    missed_valid = not meta['is_valid_solution'] and answer_included
                    missed_semiexact = not meta['semi_exact'] and answer_included
                    missed_exact = not meta['exact_solution'] and answer_included
                    stats['missed_valid_answers'][num_preds][pred_dep_percent]['found'] += 1 if missed_found else 0
                    stats['missed_valid_answers'][num_preds][pred_dep_percent]['valid'] += 1 if missed_valid else 0
                    stats['missed_valid_answers'][num_preds][pred_dep_percent]['semiexact'] += 1 if missed_semiexact else 0
                    stats['missed_valid_answers'][num_preds][pred_dep_percent]['exact'] += 1 if missed_exact else 0
                    stats['missed_valid_answers'][num_preds][pred_dep_percent]['answer_included'] += 1 if answer_included else 0

                # 3. getting accuracy stats
                stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['found'] += 1 if meta['solution_found'] else 0
                stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['valid'] += 1 if meta['is_valid_solution'] else 0
                stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['semiexact'] += 1 if meta['semi_exact'] else 0
                stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['exact'] += 1 if meta['exact_solution'] else 0
                has_some_of_targ_ans = False
                if meta['final_answer_pids']:
                    for pid in meta['final_answer_pids']:
                        if pid in meta['answer_pids']:
                            has_some_of_targ_ans = True
                            break
                stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['has_some_of_targ_ans'] += 1 if has_some_of_targ_ans else 0
                stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['valid_and_has_some_of_targ_ans'] += 1 if meta['is_valid_solution'] and has_some_of_targ_ans else 0

                # 4. extract runtime for different experiments
                stats['pred_runtime_stats'][num_preds][pred_dep_percent]['total'] += sum(meta['mip_runtimes'])
                stats['pred_runtime_stats'][num_preds][pred_dep_percent]['last'] += meta['mip_runtimes'][-1]

                # 5. final status/result
                status = meta['mip_statuses'][-1]
                is_valid = meta['is_valid_solution']
                if 'answer_pids_dep_scores_rank' in meta:
                    answer_included = meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
                else:
                    answer_included = False
                stats['final_status_result'][(status, is_valid, answer_included)] += 1

                # 6. answer dep ranks
                if 'answer_pids_dep_scores_rank' in meta:
                    stats['answer_dep_ranks'][num_preds][pred_dep_percent]['rank'].append(meta['answer_pids_dep_scores_rank'])
                    stats['answer_dep_ranks'][num_preds][pred_dep_percent]['used'].append(meta['num_dep_scores_used'])
                    stats['answer_dep_ranks'][num_preds][pred_dep_percent]['avail'].append(meta['num_dep_scores_avail'])

                # 7. statuses
                if status == 'infeasible':
                    status = 'infeasible_' + ('has_ans' if answer_included else 'no_ans')
                stats['pred_statuses'][num_preds][pred_dep_percent].append(status)

            # creating averages for accuracy & runtime stats
            stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['found'] /= float(num_runs)
            stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['valid'] /= float(num_runs)
            stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['semiexact'] /= float(num_runs)
            stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['exact'] /= float(num_runs)
            stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['has_some_of_targ_ans'] /= float(num_runs)
            stats['pred_accuracy_stats'][num_preds][pred_dep_percent]['valid_and_has_some_of_targ_ans'] /= float(num_runs)
            stats['pred_runtime_stats'][num_preds][pred_dep_percent]['total'] /= float(num_runs)
            stats['pred_runtime_stats'][num_preds][pred_dep_percent]['last'] /= float(num_runs)

    return stats


def print_pred_accuracy_results(pred_accuracy_stats, preds=range(1000, 10001, 1000),
                                pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                                keys=["found", 'valid', 'semiexact', 'exact']):
    print_accuracy_results(pred_accuracy_stats, preds, pred_dep_percents, keys)


def print_pred_runtime_results(pred_runtime_stats, preds=range(1000, 10001, 1000),
                               pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                               keys=["total", "last"]):
    print_runtime_results(pred_runtime_stats, preds, pred_dep_percents, keys)








###############################################################################
########################## Extract col results ################################

def get_col_exp_results(system, corr, num_rows, num_runs, slack_percent,
                         max_preds=2, columns=range(100, 600, 100),
                         pred_dep_percents=np.arange(0.0, 1.1, 0.1)):
    stats = {}
    stats['minos_distrib'] = defaultdict(lambda: defaultdict(int))
    stats['missed_valid_answers'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['col_accuracy_stats'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    stats['col_runtime_stats'] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for num_columns in columns:
        for pred_dep_percent in pred_dep_percents:
            for i in xrange(num_runs):
                if system in ("tiresias", "conquer"):
                    cur_sys = system
                elif system == "n_choose_k":
                    cur_sys = system[0:-1] + str(max_preds)
                elif system == "greedy":
                    cur_sys = system + str(max_preds)
                elif system == "emerilRandom":
                    cur_sys = "emDep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilIndep":
                    cur_sys = "emIndep{}".format(int(pred_dep_percent * 100))
                elif system == "emerilHybrid":
                    cur_sys = "emHybrid{}".format(int(pred_dep_percent * 100))
                random_seed = RANDOM_SEED + i
                sorting_random_seed = RANDOM_SEED + i
                data_cache_key = 'r{}-c{}-corr{}-seed{}'.format(num_rows, num_columns, corr, random_seed)
                meta_cache_key = data_cache_key + '-sys.{}-pdp{}-sort{}-sp{}'\
                                    .format(cur_sys, pred_dep_percent, sorting_random_seed, slack_percent)
                meta_cache_file = os.path.join(DATA_GEN_EXP_META_DIR, "v1", meta_cache_key + ".cache")
                with open(meta_cache_file) as f:
                    meta = pickle.load(f)

                # 1. determine which minos options work best
                for j, minos_opt_str in enumerate(meta['minos_opt_strs']):
                    status = meta['mip_statuses'][j]
                    stats['minos_distrib'][minos_opt_str][status] += 1

                # 2. compare answer dep rank vs. results
                if 'answer_pids_dep_scores_rank' in meta:
                    answer_included = meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
                    missed_found = not meta['solution_found'] and answer_included
                    missed_valid = not meta['is_valid_solution'] and answer_included
                    missed_semiexact = not meta['semi_exact'] and answer_included
                    missed_exact = not meta['exact_solution'] and answer_included
                    stats['missed_valid_answers'][num_columns][pred_dep_percent]['found'] += 1 if missed_found else 0
                    stats['missed_valid_answers'][num_columns][pred_dep_percent]['valid'] += 1 if missed_valid else 0
                    stats['missed_valid_answers'][num_columns][pred_dep_percent]['semiexact'] += 1 if missed_semiexact else 0
                    stats['missed_valid_answers'][num_columns][pred_dep_percent]['exact'] += 1 if missed_exact else 0
                    stats['missed_valid_answers'][num_columns][pred_dep_percent]['answer_included'] += 1 if answer_included else 0

                # 3. getting accuracy stats
                stats['col_accuracy_stats'][num_columns][pred_dep_percent]['found'] += 1 if meta['solution_found'] else 0
                stats['col_accuracy_stats'][num_columns][pred_dep_percent]['valid'] += 1 if meta['is_valid_solution'] else 0
                stats['col_accuracy_stats'][num_columns][pred_dep_percent]['semiexact'] += 1 if meta['semi_exact'] else 0
                stats['col_accuracy_stats'][num_columns][pred_dep_percent]['exact'] += 1 if meta['exact_solution'] else 0

                # 4. extract runtime for different experiments
                stats['col_runtime_stats'][num_columns][pred_dep_percent]['total'] += sum(meta['mip_runtimes'])
                stats['col_runtime_stats'][num_columns][pred_dep_percent]['last'] += meta['mip_runtimes'][-1]

            # creating averages for accuracy & runtime stats
            stats['col_accuracy_stats'][num_columns][pred_dep_percent]['found'] /= float(num_runs)
            stats['col_accuracy_stats'][num_columns][pred_dep_percent]['valid'] /= float(num_runs)
            stats['col_accuracy_stats'][num_columns][pred_dep_percent]['semiexact'] /= float(num_runs)
            stats['col_accuracy_stats'][num_columns][pred_dep_percent]['exact'] /= float(num_runs)
            stats['col_runtime_stats'][num_columns][pred_dep_percent]['total'] /= float(num_runs)
            stats['col_runtime_stats'][num_columns][pred_dep_percent]['last'] /= float(num_runs)

    return stats


def print_col_accuracy_results(col_accuracy_stats, columns=range(100, 600, 100),
                                pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                                keys=["found", 'valid', 'semiexact', 'exact']):
    for num_columns in columns:
        row = []
        for key in keys:
            row += [col_accuracy_stats[num_columns][pdp][key] for pdp in pred_dep_percents]
        print "\t".join(map(str, row))


def print_col_runtime_results(col_runtime_stats, columns=range(100, 600, 100),
                               pred_dep_percents=np.arange(0.0, 1.1, 0.1), keys=["total", "last"]):
    for num_columns in columns:
        row = []
        for key in keys:
            row += [col_runtime_stats[num_columns][pdp][key] for pdp in pred_dep_percents]
        print "\t".join(map(str, row))





##############################################################################
####################### generic exp result extraction ########################

def print_answer_ranks(cur_stats, keys, pred_dep_percents, num_runs,
                       debug_key=None, debug_pdp=None):
    print "\n# ANSWER RANK ANALYSIS #"
    print "## ANSWER INCLUDED COUNTS ##"
    for key in keys:
        row = []
        for pred_dep_percent in pred_dep_percents:
            ans_inc_cnt = 0
            for i in xrange(num_runs):
                rank = cur_stats['answer_dep_ranks'][key][pred_dep_percent]['rank'][i]
                used = cur_stats['answer_dep_ranks'][key][pred_dep_percent]['used'][i]
                if rank <= used:
                    ans_inc_cnt += 1
            row.append(ans_inc_cnt / float(num_runs))
        print '\t'.join(['{:.1f}'.format(x) for x in row])

    print "\n## AVG RANK in (USED / AVAIL) ##"
    for key in keys:
        row = []
        for pred_dep_percent in pred_dep_percents:
            ranks = cur_stats['answer_dep_ranks'][key][pred_dep_percent]['rank']
            used = cur_stats['answer_dep_ranks'][key][pred_dep_percent]['used']
            avail = cur_stats['answer_dep_ranks'][key][pred_dep_percent]['avail']
            if debug_key and debug_pdp and \
                    np.isclose(key, debug_key) and \
                    np.isclose(pred_dep_percent, debug_pdp):
                print "debug ranks: {}".format(ranks)
                print "debug used: {}".format(used)
                print "debug avail: {}".format(avail)
            row.append('{:.0f} in ({:.0f} / {:.0f})'.format(np.mean(ranks), np.mean(used), np.mean(avail)))
        print '     '.join(row)

    print "\n## RANKS NEEDED (min/median/max) ##"
    for key in keys:
        row = []
        for pred_dep_percent in pred_dep_percents:
            needed = []
            for i in xrange(num_runs):
                rank = cur_stats['answer_dep_ranks'][key][pred_dep_percent]['rank'][i]
                avail = cur_stats['answer_dep_ranks'][key][pred_dep_percent]['avail'][i]
                needed.append(float(rank) / avail)
            row.append('{:.2f}/{:.2f}/{:.2f}'.format(min(needed), np.median(needed), max(needed)))
        print '\t'.join(row)


def print_accuracy_results(accuracy_stats, indexes,
                           pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                           keys=["found", 'valid', 'semiexact', 'exact']):
    for index in indexes:
        row = []
        for key in keys:
            row += [accuracy_stats[index][pdp][key] for pdp in pred_dep_percents]
        print "\t".join(map(str, row))


def print_runtime_results(runtime_stats, indexes,
                          pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                          keys=["total", "last"]):
    for index in indexes:
        row = []
        for key in keys:
            row += [runtime_stats[index][pdp][key] for pdp in pred_dep_percents]
        print "\t".join(map(str, row))


def print_statuses(statuses, indexes, pred_dep_percents=np.arange(0.0, 1.1, 0.1)):
    """
    generic status printer; indexes is corrs/preds/rows
    """
    unique_statuses = set()
    for index in indexes:
        for pdp in pred_dep_percents:
            for status in statuses[index][pdp]:
                unique_statuses.add(status)

    # print counts for each cell by status
    for status in unique_statuses:
        print "## Status: {} ##".format(status)
        for index in indexes:
            row = []
            for pdp in pred_dep_percents:
                row.append(sum([1 for x in statuses[index][pdp] if x == status]))
            print "\t".join(map(str, row))





###############################################################################
####################### Extract real-world results ############################

def get_rw_exp_results(system, dataset_name, pred_mode, num_runs, slack_percent,
                       max_preds=2, pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                       rw_data_mode="rw1"):
    stats = {}
    stats['minos_distrib'] = defaultdict(lambda: defaultdict(int))
    stats['missed_valid_answers'] = defaultdict(lambda: defaultdict(int))
    stats['accuracy_stats'] = defaultdict(lambda: defaultdict(int))
    stats['runtime_stats'] = defaultdict(lambda: defaultdict(int))
    stats['final_status_result'] = defaultdict(int)
    stats['answer_dep_ranks'] = defaultdict(lambda: defaultdict(list))
    stats['statuses'] = defaultdict(list)
    for pred_dep_percent in pred_dep_percents:
        for i in xrange(num_runs):
            if system in ("tiresias", "conquer"):
                cur_sys = system
            elif system == "n_choose_k":
                cur_sys = system[0:-1] + str(max_preds)
            elif system == "greedy":
                cur_sys = system + str(max_preds)
            elif system == "emerilRandom":
                cur_sys = "emDep{}".format(int(pred_dep_percent * 100))
            elif system == "emerilIndep":
                cur_sys = "emIndep{}".format(int(pred_dep_percent * 100))
            elif system == "emerilHybrid":
                cur_sys = "emHybrid{}".format(int(pred_dep_percent * 100))
            elif system == "emerilThresRand":
                cur_sys = "emThresRand{}".format(int(pred_dep_percent * 100))
            elif system == "emerilThresRandIndep":
                cur_sys = "emThresRandIndep{}".format(int(pred_dep_percent * 100))
            elif system == "emerilThresIndep":
                cur_sys = "emThresIndep{}".format(int(pred_dep_percent * 100))
            elif system == "emerilTriHybrid":
                cur_sys = "emTriHybrid{}".format(int(pred_dep_percent * 100))
            elif system == "emerilTiHybrid":
                cur_sys = "emTrHybrid{}".format(int(pred_dep_percent * 100))
            random_seed = RANDOM_SEED + i
            sorting_random_seed = RANDOM_SEED + i
            data_cache_key = '{}-pmode.{}-seed{}'.format(dataset_name, pred_mode, random_seed)
            meta_cache_key = data_cache_key + '-sys.{}-pdp{}-sort{}-sp{}'\
                                .format(cur_sys, pred_dep_percent, sorting_random_seed, slack_percent)
            meta_cache_file = os.path.join(DATA_GEN_EXP_META_DIR, rw_data_mode, meta_cache_key + ".cache")
            with open(meta_cache_file) as f:
                meta = pickle.load(f)

            # 1. determine which minos options work best
            for j, minos_opt_str in enumerate(meta['minos_opt_strs']):
                status = meta['mip_statuses'][j]
                stats['minos_distrib'][minos_opt_str][status] += 1

            # 2. compare answer dep rank vs. results
            if 'answer_pids_dep_scores_rank' in meta:
                answer_included = meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
                missed_found = not meta['solution_found'] and answer_included
                missed_valid = not meta['is_valid_solution'] and answer_included
                missed_semiexact = not meta['semi_exact'] and answer_included
                missed_exact = not meta['exact_solution'] and answer_included
                stats['missed_valid_answers'][pred_dep_percent]['found'] += 1 if missed_found else 0
                stats['missed_valid_answers'][pred_dep_percent]['valid'] += 1 if missed_valid else 0
                stats['missed_valid_answers'][pred_dep_percent]['semiexact'] += 1 if missed_semiexact else 0
                stats['missed_valid_answers'][pred_dep_percent]['exact'] += 1 if missed_exact else 0
                stats['missed_valid_answers'][pred_dep_percent]['answer_included'] += 1 if answer_included else 0

            # 3. getting accuracy stats
            stats['accuracy_stats'][pred_dep_percent]['found'] += 1 if meta['solution_found'] else 0
            stats['accuracy_stats'][pred_dep_percent]['valid'] += 1 if meta['is_valid_solution'] else 0
            stats['accuracy_stats'][pred_dep_percent]['semiexact'] += 1 if meta['semi_exact'] else 0
            stats['accuracy_stats'][pred_dep_percent]['exact'] += 1 if meta['exact_solution'] else 0

            # 4. extract runtime for different experiments
            stats['runtime_stats'][pred_dep_percent]['total'] += sum(meta['mip_runtimes'])
            stats['runtime_stats'][pred_dep_percent]['last'] += meta['mip_runtimes'][-1]

            # 5. final status/result
            status = meta['mip_statuses'][-1]
            is_valid = meta['is_valid_solution']
            if 'answer_pids_dep_scores_rank' in meta:
                answer_included = meta['num_dep_scores_used'] and meta['answer_pids_dep_scores_rank'] <= meta['num_dep_scores_used']
            else:
                answer_included = False
            stats['final_status_result'][(status, is_valid, answer_included)] += 1

            # 6. answer dep ranks
            if 'answer_pids_dep_scores_rank' in meta:
                stats['answer_dep_ranks'][pred_dep_percent]['rank'].append(meta['answer_pids_dep_scores_rank'])
                stats['answer_dep_ranks'][pred_dep_percent]['used'].append(meta['num_dep_scores_used'])
                stats['answer_dep_ranks'][pred_dep_percent]['avail'].append(meta['num_dep_scores_avail'])

            # 7. statuses
            if status == 'infeasible':
                status = 'infeasible_' + ('has_ans' if answer_included else 'no_ans')
            stats['statuses'][pred_dep_percent].append(status)

        # creating averages for accuracy & runtime stats
        stats['accuracy_stats'][pred_dep_percent]['found'] /= float(num_runs)
        stats['accuracy_stats'][pred_dep_percent]['valid'] /= float(num_runs)
        stats['accuracy_stats'][pred_dep_percent]['semiexact'] /= float(num_runs)
        stats['accuracy_stats'][pred_dep_percent]['exact'] /= float(num_runs)
        stats['runtime_stats'][pred_dep_percent]['total'] /= float(num_runs)
        stats['runtime_stats'][pred_dep_percent]['last'] /= float(num_runs)

    return stats


def print_rw_accuracy_results(accuracy_stats, pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                              keys=["found", 'valid', 'semiexact', 'exact']):
    row = []
    for key in keys:
        row += [accuracy_stats[pdp][key] for pdp in pred_dep_percents]
    print "\t".join(map(str, row))


def print_rw_runtime_results(runtime_stats, pred_dep_percents=np.arange(0.0, 1.1, 0.1),
                             keys=["total", "last"]):
    row = []
    for key in keys:
        row += [runtime_stats[pdp][key] for pdp in pred_dep_percents]
    print "\t".join(map(str, row))


def print_rw_statuses(statuses, pred_dep_percents=np.arange(0.0, 1.1, 0.1)):
    """
    generic status printer; indexes is corrs/preds/rows
    """
    unique_statuses = set()
    for pdp in pred_dep_percents:
        for status in statuses[pdp]:
            unique_statuses.add(status)

    # print counts for each cell by status
    for status in unique_statuses:
        print "### Status: {} ###".format(status)
        row = []
        for pdp in pred_dep_percents:
            row.append(sum([1 for x in statuses[pdp] if x == status]))
        print "\t".join(map(str, row))


def print_rw_answer_ranks(cur_stats, pred_dep_percents, num_runs, debug_pdp=None):
    print "\n# ANSWER RANK ANALYSIS #"
    print "## ANSWER INCLUDED COUNTS ##"
    row = []
    for pred_dep_percent in pred_dep_percents:
        ans_inc_cnt = 0
        for i in xrange(num_runs):
            rank = cur_stats['answer_dep_ranks'][pred_dep_percent]['rank'][i]
            used = cur_stats['answer_dep_ranks'][pred_dep_percent]['used'][i]
            if rank <= used:
                ans_inc_cnt += 1
        row.append(ans_inc_cnt / float(num_runs))
    print '\t'.join(['{:.1f}'.format(x) for x in row])

    print "\n## AVG RANK in (USED / AVAIL) ##"
    row = []
    for pred_dep_percent in pred_dep_percents:
        ranks = cur_stats['answer_dep_ranks'][pred_dep_percent]['rank']
        used = cur_stats['answer_dep_ranks'][pred_dep_percent]['used']
        avail = cur_stats['answer_dep_ranks'][pred_dep_percent]['avail']
        if debug_pdp and np.isclose(pred_dep_percent, debug_pdp):
            print "debug ranks: {}".format(ranks)
            print "debug used: {}".format(used)
            print "debug avail: {}".format(avail)
        row.append('{:.0f} in ({:.0f} / {:.0f})'.format(np.mean(ranks), np.mean(used), np.mean(avail)))
    print '     '.join(row)

    print "\n## RANKS NEEDED (median/max) ##"
    row = []
    for pred_dep_percent in pred_dep_percents:
        needed = []
        for i in xrange(num_runs):
            rank = cur_stats['answer_dep_ranks'][pred_dep_percent]['rank'][i]
            avail = cur_stats['answer_dep_ranks'][pred_dep_percent]['avail'][i]
            needed.append(float(rank) / avail)
        row.append('{:.2f}/{:.2f}'.format(np.median(needed), max(needed)))
    print '\t'.join(row)




############################# answer eval #####################################

def verify_preds_answer(df, preds, target_fld, target_counts, adjusted_bin_edges,
                        slack_percent, answer_pids):
    answer_preds = [preds[pid] for pid in answer_pids]
    answer_df = get_df_from_many_preds(df, answer_preds)
    answer_distrib = np.histogram(answer_df[target_fld], adjusted_bin_edges)[0]
    target_bounds = get_target_count_bounds(target_counts, slack_percent)
    is_valid = True
    for bin_id, (lower, upper) in enumerate(target_bounds):
        if answer_distrib[bin_id] < lower or answer_distrib[bin_id] > upper:
            is_valid = False
            break
    signal_dist = np.sum(np.abs(answer_distrib - target_counts))
    print "is valid: {}, signal dist: {}".format(is_valid, signal_dist)
    for bin_id, (lower, upper) in enumerate(target_bounds):
        in_bounds = (lower <= answer_distrib[bin_id] <= upper)
        print "in_bound={}, {} <= {} <= {}".format(in_bounds, lower, answer_distrib[bin_id], upper)
