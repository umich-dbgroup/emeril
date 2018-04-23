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
from .candidate_gen import generate_preds_v5
from .syndata import (
    get_synthetic_answer_v2, get_synthetic_answer_v3,
    get_target_fld_bin_vals, get_target_fld_vals_from_df,
    update_meta_with_user_sims, thres_rand_dep_filter,
    wdi_get_target_fld_vals_from_df
)
from .mipsolver import (
    get_pred_pair_meta, get_dep_scores_v2, run_mip_solver_v2,
    run_emeril_hybrid_v1, update_pred_pair_meta_with_new_preds
)
from .utils import bounded_normal_draw

###############################################################################
###############################################################################
##################### Real World Data Experiments #############################
###############################################################################
###############################################################################





###############################################################################
############################## Data loading ###################################

def get_real_world_data(dataset_name, print_details=False):
    if dataset_name == "nhanes":
        data_dir = '/z/dol/emeril-real-world-data/nhanes'
        demographic_df = pd.read_csv(os.path.join(data_dir, 'demographic.csv'))
        clean_df_column_names(demographic_df)

        diet_df = pd.read_csv(os.path.join(data_dir, 'diet.csv'))
        clean_df_column_names(diet_df)

        examination_df = pd.read_csv(os.path.join(data_dir, 'examination.csv'))
        clean_df_column_names(examination_df)

        labs_df = pd.read_csv(os.path.join(data_dir, 'labs.csv'))
        clean_df_column_names(labs_df)

        medications_df = pd.read_csv(os.path.join(data_dir, 'medications.csv'))
        clean_df_column_names(medications_df)

        questionnaire_df = pd.read_csv(os.path.join(data_dir, 'questionnaire.csv'))
        clean_df_column_names(questionnaire_df)

        ## creating combined dataframes ##
        dem_exam_df = examination_df.merge(demographic_df, on='SEQN', how='left')
        dem_exam_lab_df = dem_exam_df.merge(labs_df, on='SEQN', how='left')
        df = dem_exam_lab_df

    elif dataset_name == "wdi":
        data_dir = '/z/dol/emeril-real-world-data/wdi'
        wdi_df = pd.read_csv(os.path.join(data_dir, 'WDIData.csv'), low_memory=False)
        clean_df_column_names(wdi_df)
        df = wdi_df

    elif dataset_name == "gtd":
        data_dir = '/z/dol/emeril-real-world-data/gtd'
        gtd_df = pd.read_csv(os.path.join(data_dir, 'globalterrorismdb_0617dist.csv'), low_memory=False)
        clean_df_column_names(gtd_df)
        df = gtd_df

    elif dataset_name == "atus":
        data_dir = '/z/dol/emeril-real-world-data/atus'
        atus_df = pd.read_table(os.path.join(data_dir, '34453-0004-Data.tsv'), low_memory=False)
        clean_df_column_names(atus_df)
        df = atus_df

    elif dataset_name == "food":
        data_dir = '/z/dol/emeril-real-world-data/food-facts'
        food_df = pd.read_csv(os.path.join(data_dir, 'FoodFacts.csv'), low_memory=False)
        clean_df_column_names(food_df)
        df = food_df

    elif dataset_name == "nfl":
        data_dir = '/z/dol/emeril-real-world-data/nfl'
        nfl_df = pd.read_csv(os.path.join(data_dir, 'NFLPlaybyPlay2015.csv'), low_memory=False)
        clean_df_column_names(nfl_df)
        df = nfl_df


    db_meta = DatabaseMeta()
    db_meta.analyze_dfs({dataset_name: df})

    if print_details:
        print("{}: num_rows={}, num_columns={}".format(dataset_name, len(df), len(df.columns)))
        db_meta.print_summary()
    return df, db_meta









###############################################################################
######################## Adding synthetic answers #############################

def add_nhanes_syn_answer_v1(df, random_seed, print_details=False):
    """
    ## OLD NOTES: ##
    Here we modify the NHANES dataset so that a filter on blood pressure changes race.

    **General goals of Use Case 1:**
    - PREDS:  p1, ..., pk
    - EMERIL: non-pred field changed: c1
    - USER:   c1.dist = *signal*
    - EMERIL: new preds

    **NHANES Specifics:**
    - story: reproducing, newer dataset
    - preds: blood pressure (BPXSY1) > 150, cholesterol (LBXSCH) > 300
    - case1.c1: race changed (RIDRETH3)
    - case1.c1.dist: pressure > 50
    - case1.c1.target_signal: uniform?
    """
    df = copy.deepcopy(df)

    # 1. set dataset specifics
    target_fld = 'RIDRETH3'  # race
    ans1_fld = 'BPXSY1'  # blood pressure
    ans2_fld = 'LBXSCH'  # cholesterol

    # 2. data cleanup (deal with nulls in target_fld)
    np_random = np.random.RandomState(random_seed)
    notna_vals = df[~np.isnan(df[target_fld])][target_fld].unique()
    for i in df[np.isnan(df[target_fld])].index:
        df.ix[i, target_fld] = notna_vals[np_random.randint(0, len(notna_vals))]

    return df, target_fld, ans1_fld, ans2_fld


def add_food_syn_answer_v1(df, random_seed, print_details=False):
    # 1. set dataset specifics
    target_fld = 'carbohydrates_100g'  # carbs
    ans1_fld = 'cholesterol_100g'  # cholesterol
    ans2_fld = 'sugars_100g'  # sugar

    # 2. data cleanup (deal with nulls in target_fld)
    np_random = np.random.RandomState(random_seed)
    notna_vals = df[~np.isnan(df[target_fld])][target_fld].unique()
    for i in df[np.isnan(df[target_fld])].index:
        df.ix[i, target_fld] = notna_vals[np_random.randint(0, len(notna_vals))]

    return df, target_fld, ans1_fld, ans2_fld


def add_wdi_syn_answer_v1(df, random_seed, print_details=False):
    # 1. set dataset specifics
    target_fld = 'c_1990'  #
    ans1_fld = 'c_1980'  #
    ans2_fld = 'c_1970'  #

    # 2. data cleanup (deal with nulls in target_fld)
    np_random = np.random.RandomState(random_seed)
    notna_vals = df[~np.isnan(df[target_fld])][target_fld].unique()
    for i in df[np.isnan(df[target_fld])].index:
        df.ix[i, target_fld] = notna_vals[np_random.randint(0, len(notna_vals))]

    return df, target_fld, ans1_fld, ans2_fld


def add_gtd_syn_answer_v1(df, random_seed, print_details=False):
    # 1. set dataset specifics
    target_fld = 'iyear'  # year
    ans1_fld = 'country'  #
    ans2_fld = 'region'  #

    # 2. data cleanup (deal with nulls in target_fld)
    np_random = np.random.RandomState(random_seed)
    notna_vals = df[~np.isnan(df[target_fld])][target_fld].unique()
    for i in df[np.isnan(df[target_fld])].index:
        df.ix[i, target_fld] = notna_vals[np_random.randint(0, len(notna_vals))]

    return df, target_fld, ans1_fld, ans2_fld


def add_nfl_syn_answer_v1(df, random_seed, print_details=False):
    # 1. set dataset specifics
    target_fld = 'ScoreDiff'  #
    ans1_fld = 'Season'  #
    ans2_fld = 'Penalty_Yards'  #

    # 2. data cleanup (deal with nulls in target_fld)
    np_random = np.random.RandomState(random_seed)
    notna_vals = df[~np.isnan(df[target_fld])][target_fld].unique()
    for i in df[np.isnan(df[target_fld])].index:
        df.ix[i, target_fld] = notna_vals[np_random.randint(0, len(notna_vals))]

    return df, target_fld, ans1_fld, ans2_fld


def add_atus_syn_answer_v1(df, random_seed, print_details=False):
    # 1. set dataset specifics
    target_fld = 'PRTAGE'  # age
    ans1_fld = 'HEFAMINC'  # family income
    ans2_fld = 'HRNUMHOU'  # num people in household

    # 2. data cleanup (deal with nulls in target_fld)
    np_random = np.random.RandomState(random_seed)
    notna_vals = df[~np.isnan(df[target_fld])][target_fld].unique()
    for i in df[np.isnan(df[target_fld])].index:
        df.ix[i, target_fld] = notna_vals[np_random.randint(0, len(notna_vals))]

    return df, target_fld, ans1_fld, ans2_fld


def add_real_world_syn_answer_v1(dataset_name, df, random_seed, print_details=False):
    df = copy.deepcopy(df)
    pre_syn_answer_funcs = {
        'wdi': add_wdi_syn_answer_v1,
        'gtd': add_gtd_syn_answer_v1,
        'food': add_food_syn_answer_v1,
        'nfl': add_nfl_syn_answer_v1,
        'atus': add_atus_syn_answer_v1,
        'nhanes': add_nhanes_syn_answer_v1,
    }
    pre_syn_answer_func = pre_syn_answer_funcs[dataset_name]
    df, target_fld, ans1_fld, ans2_fld = \
            pre_syn_answer_func(df, random_seed, print_details=print_details)

    # sorting df by target_fld since answer generation assumes this
    df.sort_values(by=[target_fld])

    # 3. select target signal from column (nhanes: normal distrib race -> uniform)
    target_fld_bin_vals, bin_counts, adjusted_bin_edges = \
            get_target_fld_vals_from_df(df, target_fld, random_seed)

    # 4. generate synthetic answer values
    target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, answer_preds, noise_preds = \
            get_synthetic_answer_v2(df, target_fld_bin_vals, bin_counts, print_details=print_details)

    # 5. replace answer columns (nhanes: blood pressure and cholesterol)
    df[target_fld] = target_fld_vals
    df[ans1_fld] = answer1_vals
    df[ans2_fld] = answer2_vals
    answer_preds[0] = (ans1_fld, answer_preds[0][1], answer_preds[0][2])
    answer_preds[1] = (ans2_fld, answer_preds[1][1], answer_preds[1][2])

    return df, target_fld, target_counts, adjusted_bin_edges, bin_counts, answer_preds



###############################################################################
########################### Experiment running ################################

def get_real_world_data_and_answer_v1(dataset_name, random_seed, data_cache_file,
                                      pred_mode='naive', print_details=False,
                                      code_sim=False):
    """
    Creates df, preds, synthetic answer for real-world dataset
    """

    if code_sim:
        raise Exception("NOT IMPLEMENTED!!")


    if print_details:
        print("Loading real world data and answer (v1)...")
    if os.path.exists(data_cache_file):
        if print_details:
            print("{} - Loading via cache.".format(dt.datetime.now()))
        with open(data_cache_file) as f:
            df, preds, meta = pickle.load(f)
    else:
        start = timer()
        np_random = np.random.RandomState(random_seed)

        # 1. getting dataset
        if print_details:
            print("{} - Getting dataset...".format(dt.datetime.now()))
        df, db_meta = get_real_world_data(dataset_name, print_details=print_details)

        # 2. adding synthetic answer to df
        if print_details:
            print("{} - Adding synthetic answer...".format(dt.datetime.now()))
        df, target_fld, target_counts, adjusted_bin_edges, bin_counts, answer_preds = \
                add_real_world_syn_answer_v1(dataset_name, df, random_seed, print_details=print_details)

        # 3. generate preds
        if print_details:
            print("{} - Generating preds (mode={})...".format(dt.datetime.now(), pred_mode))
        if pred_mode == 'naive':
            preds = generate_preds_v5(df, db_meta, target_fld)
        elif pred_mode == 'standard':
            raise Exception("Not implemented")
        elif pred_mode == 'aggressive':
            raise Exception("Not implemented")
        else:
            raise Exception("Invalid pred_mode!")

        # 4. finding answer_preds in preds / adding to preds
        answer_pids = []
        found_pred = [False, False]
        for pid, pred in enumerate(preds):
            for ans_pid in xrange(len(answer_preds)):
                if not found_pred[ans_pid] and pred == answer_preds[ans_pid]:
                    found_pred[ans_pid] = True
                    answer_pids.append(pid)
                    if print_details:
                        print "- found answer[{}], pid={}".format(ans_pid, pid)
        for ans_pid in xrange(len(answer_preds)):
            if not found_pred[ans_pid]:
                preds.append(answer_preds[ans_pid])
                pid = len(preds) - 1
                answer_pids.append(pid)
                if print_details:
                    print "- didn't find answer[{}], manually added (pid={})".format(ans_pid, pid)

        # 5. noting some meta data
        meta = {}
        meta['target_fld'] = target_fld
        meta['target_counts'] = target_counts
        meta['adjusted_bin_edges'] = adjusted_bin_edges
        meta['bin_counts'] = np.histogram(df[target_fld], adjusted_bin_edges)[0]
        meta['answer_preds'] = answer_preds
        meta['answer_pids'] = answer_pids
        meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)
        qs = Query.get_pandas_query_from_preds(answer_preds)
        meta['answer_counts'] = np.histogram(df.query(qs)[meta['target_fld']], meta['adjusted_bin_edges'])[0]

        # 6. caching results
        if print_details:
            print("{} - Saving to cache...".format(dt.datetime.now()))
        meta['data_gen_runtime'] = timer() - start
        with open(data_cache_file, "w") as f:
            pickle.dump((df, preds, meta), f, -1)

    return df, preds, meta


def update_meta_with_rw2_data(dataset_name, df, meta, preds):
    v2_answer_pids = {
        'nhanes': [22, 8811],
        'wdi': [301, 304],
        'gtd': [23, 133],
        'atus': [22, 154],
        'food': [41, 575],
        'nfl': [2, 427],
    }
    answer_pids = v2_answer_pids[dataset_name]
    answer_preds = [preds[answer_pids[0]], preds[answer_pids[1]]]
    meta['answer_preds'] = answer_preds
    meta['answer_pids'] = answer_pids
    meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)
    qs = Query.get_pandas_query_from_preds(answer_preds)
    meta['answer_counts'] = np.histogram(df.query(qs)[meta['target_fld']], meta['adjusted_bin_edges'])[0]
    meta['target_counts'] = meta['answer_counts']


def test_rw3_updating(df, meta, preds, random_seed,
                      percent_non_answer_used=0.7,
                      ans1_percent_non_answer_used="normal",
                      noise_percent_answer_used=0.9,
                      noise_percent_non_answer_used=0.1):

    # getting target vals
    target_fld_bin_vals, bin_counts, adjusted_bin_edges = \
            get_target_fld_vals_from_df(df, meta['target_fld'], random_seed)

    # defining params
    if percent_non_answer_used == "normal":
        percent_non_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.2, min_val=0.1, max_val=0.9)
    if ans1_percent_non_answer_used == "normal":
        ans1_percent_non_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.2, min_val=0.1, max_val=0.9)
    if noise_percent_answer_used == "normal":
        noise_percent_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.2, min_val=0.1, max_val=0.9)
    if noise_percent_non_answer_used == "normal":
        noise_percent_non_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.2, min_val=0.1, max_val=0.9)

    # getting syn answer
    np_random = np.random.RandomState(random_seed)
    target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, answer_preds, noise_preds = \
            get_synthetic_answer_v3(df, target_fld_bin_vals, bin_counts, print_details=False,
                                    percent_non_answer_used=percent_non_answer_used,
                                    ans1_percent_non_answer_used=ans1_percent_non_answer_used,
                                    noise_percent_answer_used=noise_percent_answer_used,
                                    noise_percent_non_answer_used=noise_percent_non_answer_used)

    # adding additional columns, updating answer preds
    df = copy.deepcopy(df)
    df['answer1'] = answer1_vals
    df['answer2'] = answer2_vals
    df['noise1'] = noise1_vals

    preds = copy.deepcopy(preds)
    del preds[-2:]
    preds += answer_preds + noise_preds

    print "full: ", list(np.histogram(df[meta['target_fld']], meta['adjusted_bin_edges'])[0])
    print "target: ", list(meta['target_counts']), "\n"

    for pid in xrange(len(preds) - 3, len(preds)):
        d = np.histogram(df.query(Query.get_qs_pred(*preds[pid]))[meta['target_fld']], meta['adjusted_bin_edges'])[0]
        #print d
        signal_dist = np.sum(np.abs(meta['target_counts'] - d))
        print "pid={}, corr={:.3f}, signal_dist={:.0f}, pred: {}\n"\
            .format(pid, pearsonr(meta['target_counts'], d)[0], signal_dist, preds[pid])

    d = np.histogram(df.query(Query.get_pandas_query_from_preds(answer_preds))[meta['target_fld']], adjusted_bin_edges)[0]
    #print d
    signal_dist = np.sum(np.abs(meta['target_counts'] - d))
    print "Combo corr={:.3f}, signal_dist={:.0f}, pred: {}\n"\
        .format(pearsonr(meta['target_counts'], d)[0], signal_dist, answer_preds)


def update_meta_with_rw3_data(rw_data_mode, df, meta, preds, random_seed,
                              pred_pair_counts, bin_pred_probs, bin_pred_pair_probs,
                              needs_wdi_fix=False):

    # since WDI has outlier issue with target field, update bins/etc.
    if needs_wdi_fix:
        target_fld_bin_vals, bin_counts, adjusted_bin_edges = \
                wdi_get_target_fld_vals_from_df(df, meta['target_fld'], random_seed)
    else:
        target_fld_bin_vals, bin_counts, adjusted_bin_edges = \
                get_target_fld_vals_from_df(df, meta['target_fld'], random_seed)


    np_random = np.random.RandomState(random_seed)
    if rw_data_mode == "rw3":
        percent_non_answer_used = 0.7  # this is the percent of rows outside of answer tuples to throw off individ corr
        ans1_percent_non_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.2, min_val=0.1, max_val=0.9)  # was 0.5
        noise_percent_answer_used = 0.9 # percent of answer and non-answer used by noise column
        noise_percent_non_answer_used = 0.1
    elif rw_data_mode == "rw4":  # makes
        percent_non_answer_used = 0.9  # this is the percent of rows outside of answer tuples to throw off individ corr
        ans1_percent_non_answer_used = 0.5
        noise_percent_answer_used = 0.8 # percent of answer and non-answer used by noise column
        noise_percent_non_answer_used = 0.2
    target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, answer_preds, noise_preds = \
            get_synthetic_answer_v3(df, target_fld_bin_vals, bin_counts, print_details=False,
                                    percent_non_answer_used=percent_non_answer_used,
                                    ans1_percent_non_answer_used=ans1_percent_non_answer_used,
                                    noise_percent_answer_used=noise_percent_answer_used,
                                    noise_percent_non_answer_used=noise_percent_non_answer_used)

    df[meta['target_fld']] = target_fld_vals

    # . adding additional columns, changing answer columns
    df['answer1'] = answer1_vals
    df['answer2'] = answer2_vals
    df['noise1'] = noise1_vals
    del preds[-2:]  # removing old synthethic-answer preds
    preds += answer_preds + noise_preds
    meta['answer_preds'] = answer_preds
    meta['answer_pids'] = [len(preds) - 3, len(preds) - 2]
    meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)
    qs = Query.get_pandas_query_from_preds(answer_preds)
    meta['answer_counts'] = target_counts #np.histogram(df.query(qs)[meta['target_fld']], meta['adjusted_bin_edges'])[0]
    meta['target_counts'] = target_counts #meta['answer_counts']

    print "Pred debugging:"
    for pid in xrange(len(preds) - 6, len(preds)):
        d = np.histogram(df.query(Query.get_qs_pred(*preds[pid]))[meta['target_fld']], meta['adjusted_bin_edges'])[0]
        signal_dist = np.sum(np.abs(meta['target_counts'] - d))
        print("- pid={}, corr={:.3f}, signal_dist={:.0f}, pred: {}"
              .format(pid, pearsonr(meta['target_counts'], d)[0], signal_dist, preds[pid]))

    # updating pred pair meta
    if pred_pair_counts:
        new_pred_ids = [len(preds) - 3, len(preds) - 2, len(preds) - 1]
        update_pred_pair_meta_with_new_preds(new_pred_ids, df, preds, meta, bin_counts,
                                             target_fld_vals, pred_pair_counts, bin_pred_probs,
                                             bin_pred_pair_probs)


def fix_wdi_distrib(df, meta, preds, random_seed, print_details=False):
    # 3. select target signal from column (nhanes: normal distrib race -> uniform)
    target_fld_bin_vals, bin_counts, adjusted_bin_edges = \
            wdi_get_target_fld_vals_from_df(df, meta['target_fld'], random_seed)

    # 4. generate synthetic answer values
    target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, answer_preds, noise_preds = \
            get_synthetic_answer_v2(df, target_fld_bin_vals, bin_counts, print_details=print_details)

    df['answer3'] = answer1_vals
    df['answer4'] = answer2_vals
    df[meta['target_fld']] = target_fld_vals
    answer_preds[0] = ('answer3', answer_preds[0][1], answer_preds[0][2])
    answer_preds[1] = ('answer4', answer_preds[1][1], answer_preds[1][2])

    meta['adjusted_bin_edges'] = adjusted_bin_edges
    meta['bin_counts'] = np.histogram(df[meta['target_fld']], adjusted_bin_edges)[0]
    meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)
    qs = Query.get_pandas_query_from_preds(answer_preds)
    meta['answer_counts'] = np.histogram(df.query(qs)[meta['target_fld']], meta['adjusted_bin_edges'])[0]
    meta['target_counts'] = target_counts #meta['answer_counts']

    preds += answer_preds
    meta['answer_preds'] = answer_preds
    meta['answer_pids'] = [len(preds) - 3, len(preds) - 2]
    meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)


def get_real_world_data_and_pred_pair_meta(dataset_name, random_seed, system,
                                           pred_mode='naive', print_details=False,
                                           min_print_details=True, n_processes=1,
                                           skip_loading=False, rw_data_mode="rw1",
                                           use_code_sim=False):
    """
    loads real world dataset from csv, builds preds, and pred-pair meta
    """
    # 1. load/generate synthetic data with synthetic answer
    start = timer()
    data_cache_key = '{}-pmode.{}-seed{}'.format(dataset_name, pred_mode, random_seed)
    data_cache_file = os.path.join(DATA_GEN_CACHE_DIR, "rw1", data_cache_key + ".cache")
    df, preds, meta = get_real_world_data_and_answer_v1(dataset_name, random_seed,
                                                        data_cache_file,
                                                        pred_mode=pred_mode,
                                                        print_details=print_details)
    if min_print_details or print_details:
        print("RUNTIME: load/generate synthetic data (rw_data_mode={}, {} rows, {} cols, {} preds): {:.3f} sec"\
                .format(rw_data_mode, len(df), len(df.columns), len(preds), timer() - start))

    # 2. load/generate dep_score meta, pred/pred-pair probs
    if system in ('tiresias', 'emerilRandom', 'emerilIndep', 'emerilHybrid', "emerilThresRand", "emerilThresRandIndep", 'emerilTriHybrid', "emerilTiHybrid", 'emerilThresIndep'):
        start = timer()
        pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs = \
            get_pred_pair_meta(df, preds, meta['target_fld'], meta['adjusted_bin_edges'],
                               "rw1", data_cache_key, print_details=print_details,
                               n_processes=n_processes, skip_loading=skip_loading)
        if min_print_details or print_details:
            print("RUNTIME: load/generate dep_score meta: {:.3f} sec ({} procs)"
                  .format(timer() - start, n_processes))
    else:
        pred_pair_counts = None
        bin_probs = None
        bin_pred_probs = None
        bin_pred_pair_probs = None
        if min_print_details or print_details:
            print("SKIPPING pred_pair_counts, bin_probs, etc. for system ({})".format(system))

    # 3. switching synthetic answer if rw_data_mode == "rw1"
    if rw_data_mode == "rw1":
        # fixing WDI data distrib issue (due to outliers w/ data)
        if dataset_name == 'wdi':
            fix_wdi_distrib(df, meta, preds, random_seed)
        pass  # we leave answers as is
    elif rw_data_mode == "rw2":
        update_meta_with_rw2_data(dataset_name, df, meta, preds)
    elif rw_data_mode in ("rw3", "rw4"):
        start = timer()
        if dataset_name == 'wdi':
            needs_wdi_fix = True
        else:
            needs_wdi_fix = False
        update_meta_with_rw3_data(rw_data_mode, df, meta, preds, random_seed,
                                  pred_pair_counts, bin_pred_probs, bin_pred_pair_probs,
                                  needs_wdi_fix=needs_wdi_fix)
        if min_print_details or print_details:
            print("RUNTIME: rw3 data update ({} preds): {:.3f} sec".format(len(preds), timer() - start))
    else:
        raise Exception("bad data mode")

    # 4. if code sim, adding more answers and preds
    if use_code_sim:
        update_meta_with_user_sims(syn_data_mode, df, meta, preds,
                                   pred_pair_counts, bin_pred_probs, bin_pred_pair_probs)

    return df, preds, meta, pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs


def test_real_world_v1(dataset_name, random_seed, pred_dep_percent,
                       print_details=False, mip_solver_timeout=600,
                       dep_scoring_method=None, dep_sorting_method=None,
                       min_meta=True, ignore_meta_cache=False, sorting_random_seed=None,
                       slack_percent=0.2, minos_options=None, system="emerilRandom",
                       pred_mode='naive', max_preds=2, min_print_details=True,
                       solver='minos', rw_data_mode="rw1",
                       add_timeout_to_meta_name=False, use_code_sim=False):
    """
    Does data gen and mip solver testing w/ provided params
    """
    if dep_sorting_method is not None:
        raise Exception("Using deprecated parameter (dep sorting).")
    if dep_scoring_method is not None:
        raise Exception("Using deprecated parameter (depscoring).")

    if system in ("tiresias", "conquer"):
        cur_sys = system
    elif system == "n_choose_k":
        cur_sys = system[0:-1] + str(max_preds)
    elif system == "greedy":
        cur_sys = system + str(max_preds)
    elif system == "emerilRandom":
        dep_sorting_method = "random"
        cur_sys = "emDep{}".format(int(pred_dep_percent * 100))
    elif system == "emerilIndep":
        dep_scoring_method = "independence"
        dep_sorting_method = "scores"
        cur_sys = "emIndep{}".format(int(pred_dep_percent * 100))
    elif system == "emerilHybrid":
        dep_scoring_method = "independence"
        dep_sorting_method = "scores"
        cur_sys = "emHybrid{}".format(int(pred_dep_percent * 100))
    elif system == "emerilThresRand":
        dep_scoring_method = "independence"
        dep_sorting_method = "scores"
        cur_sys = "emThresRand{}".format(int(pred_dep_percent * 100))
    elif system == "emerilThresRandIndep":
        dep_scoring_method = "independence"
        dep_sorting_method = "scores"
        cur_sys = "emThresRandIndep{}".format(int(pred_dep_percent * 100))
    elif system == "emerilThresIndep":
        dep_scoring_method = "independence"
        dep_sorting_method = "scores"
        cur_sys = "emThresIndep{}".format(int(pred_dep_percent * 100))
    elif system == "emerilTriHybrid":
        dep_scoring_method = "independence"
        dep_sorting_method = "scores"
        cur_sys = "emTriHybrid{}".format(int(pred_dep_percent * 100))
    elif system == "emerilTiHybrid":
        dep_scoring_method = "independence"
        dep_sorting_method = "scores"
        cur_sys = "emTrHybrid{}".format(int(pred_dep_percent * 100))

    data_cache_key = '{}-pmode.{}-seed{}'.format(dataset_name, pred_mode, random_seed)
    sorting_random_seed = sorting_random_seed if sorting_random_seed else random_seed
    meta_cache_key = data_cache_key + '-sys.{}-pdp{}-sort{}-sp{}'\
                        .format(cur_sys, pred_dep_percent, sorting_random_seed, slack_percent)
    if add_timeout_to_meta_name:
        meta_cache_key += '-time{}'.format(mip_solver_timeout)
    if use_code_sim:
        meta_cache_key += '-codesim'
    meta_cache_file = os.path.join(DATA_GEN_EXP_META_DIR, rw_data_mode, meta_cache_key + ".cache")
    if not ignore_meta_cache and os.path.exists(meta_cache_file):
        with open(meta_cache_file) as f:
            meta = pickle.load(f)
    else:
        # 1. getting data
        df, preds, meta, pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs = \
                get_real_world_data_and_pred_pair_meta(dataset_name, random_seed, system,
                                                       pred_mode=pred_mode,
                                                       print_details=print_details,
                                                       min_print_details=min_print_details,
                                                       rw_data_mode=rw_data_mode,
                                                       use_code_sim=use_code_sim)
        if 'user_pred_ids' not in meta:
            meta['user_pred_ids'] = None
        if print_details:
            target_bounds = get_target_count_bounds(meta['target_counts'], slack_percent)
            print "target bounds: {}".format(target_bounds)
        if min_print_details or print_details:
            print "target counts:", meta['target_counts']
            print "answer pids: {}".format(meta['answer_pids'])
            print "answer query:", meta['answer_query']
            print "answer counts:", meta['answer_counts']

        # 2. get dep scores for experiment params (dep score func & sorting, pred_dep_percent)
        if dep_sorting_method is not None:
            start = timer()
            np_random = np.random.RandomState(sorting_random_seed)
            dep_scores_pdp = 1.0 if system in ("emerilThresRand", "emerilThresRandIndep", "emerilTriHybrid", "emerilTiHybrid", 'emerilThresIndep') else pred_dep_percent
            dep_scores, ds_meta = get_dep_scores_v2(np_random, pred_pair_counts,
                                                    len(df), meta['answer_pids'],
                                                    scoring_method=dep_scoring_method,
                                                    sorting_method=dep_sorting_method,
                                                    pred_dep_percent=dep_scores_pdp,
                                                    print_details=print_details)
            meta.update(ds_meta)
            if min_print_details or print_details:
                print("RUNTIME: get dep scores for experiment params: {:.3f} sec".format(timer() - start))
        else:
            if min_print_details or print_details:
                print("SKIPPING dep_scores since cur system ({}) doesn't use.".format(system))

        # 3. if system == emerilThresRand, filter dep scores accordingly
        if system in ("emerilThresRand", "emerilThresRandIndep", "emerilTriHybrid", "emerilTiHybrid", 'emerilThresIndep'):
            start = timer()
            dep_scores = thres_rand_dep_filter(df, meta, dep_scores, bin_pred_probs,
                                               bin_pred_pair_probs, pred_pair_counts,
                                               pred_dep_percent, sorting_random_seed,
                                               slack_percent, system)
            if min_print_details or print_details:
                print("RUNTIME: thres-rand filtering: {:.3f} sec".format(timer() - start))

        # 4. run mip for experiment params (dep score func & sorting, pred_dep_percent)
        start = timer()
        if system == "tiresias":
            meta.update(run_tiresias_solver_v1(df, preds, meta['target_counts'],
                                               meta['target_fld'], meta['adjusted_bin_edges'], meta['answer_pids'],
                                               slack_percent=slack_percent, slack_increment=None, slack_max=1.0,
                                               mip_solver_timeout=mip_solver_timeout,
                                               print_details=print_details,
                                               filename_uid=meta_cache_key,
                                               syn_data_mode=rw_data_mode))
        elif system == "conquer":
            meta.update(run_conquer_solver_v1(df, preds, meta['target_counts'],
                                              meta['target_fld'], meta['adjusted_bin_edges'], meta['answer_pids'],
                                              print_details=print_details))
        elif system == "n_choose_k":
            meta.update(run_n_choose_k_solver_v1(df, preds, meta['target_counts'], meta['target_fld'],
                                                 meta['adjusted_bin_edges'], meta['answer_pids'],
                                                 slack_percent=slack_percent,
                                                 print_details=print_details,
                                                 max_preds=max_preds,
                                                 sorting_random_seed=sorting_random_seed,
                                                 timeout=mip_solver_timeout))
        elif system == "greedy":
            meta.update(run_greedy_solver_v1(df, preds, meta['target_counts'], meta['target_fld'],
                                             meta['adjusted_bin_edges'], meta['answer_pids'],
                                             slack_percent=slack_percent,
                                             print_details=print_details,
                                             max_preds=max_preds))
        elif system in ("emerilHybrid", "emerilTriHybrid", "emerilTiHybrid"):
            meta.update(run_emeril_hybrid_v1(df, preds, dep_scores, bin_probs, bin_pred_probs,
                                             bin_pred_pair_probs, meta['target_counts'],
                                             meta['target_fld'], meta['adjusted_bin_edges'], meta['answer_pids'],
                                             slack_percent=slack_percent,
                                             max_preds=max_preds,
                                             slack_increment=None, slack_max=1.0,
                                             mip_solver_timeout=mip_solver_timeout,
                                             print_details=print_details,
                                             filename_uid=meta_cache_key,
                                             syn_data_mode=rw_data_mode,
                                             minos_options=minos_options,
                                             min_print_details=min_print_details,
                                             solver=solver,
                                             user_pred_ids=meta['user_pred_ids']))

        # otherwise, assuming emeril
        else:
            meta.update(run_mip_solver_v2(df, preds, dep_scores, bin_probs, bin_pred_probs,
                                          bin_pred_pair_probs, meta['target_counts'],
                                          meta['target_fld'], meta['adjusted_bin_edges'], meta['answer_pids'],
                                          slack_percent=slack_percent, slack_increment=None, slack_max=1.0,
                                          mip_solver_timeout=mip_solver_timeout,
                                          print_details=print_details,
                                          filename_uid=meta_cache_key,
                                          syn_data_mode=rw_data_mode,
                                          minos_options=minos_options,
                                          min_print_details=min_print_details,
                                          solver=solver,
                                          user_pred_ids=meta['user_pred_ids']))
        if min_print_details or print_details:
            print("RUNTIME: solving problem with cur system: {:.3f} sec".format(timer() - start))

        # 5. saving meta cache
        with open(meta_cache_file, "w") as f:
            pickle.dump(meta, f, -1)

    return meta
