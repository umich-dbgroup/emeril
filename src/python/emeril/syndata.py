from . import *
from .mipsolver import (
    get_pred_pair_meta, get_dep_scores_v2, run_mip_solver_v2,
    run_tiresias_solver_v1, run_emeril_hybrid_v1,
    update_pred_pair_meta_with_new_preds
)
from .utils import get_adjusted_bin_edges, print_avg_column_corr, bounded_normal_draw
from .baselines import (
    run_freq_mining_solver_v1, run_n_choose_k_solver_v1,
    run_greedy_solver_v1
)
from .utils import get_target_count_bounds

###############################################################################
###############################################################################
###################### Synthetic Data Generation ##############################
###############################################################################
###############################################################################







###############################################################################
######################### Utility/Debug Functions #############################

######################### helper functions used in rw #########################



def wdi_get_target_fld_vals_from_df(df, target_fld, random_seed=RANDOM_SEED, debug=False):
    """
    generates binned values for target field
    """
    # 1. shuffling target_fld values
    np_random = np.random.RandomState(random_seed)
    target_fld_vals = list(df[target_fld])
    np_random.shuffle(target_fld_vals)

    # 2. getting bin_edges and bin_counts
    unique_target_fld_vals = df[target_fld].unique()
    if len(unique_target_fld_vals) <= 10:
        bin_edges = sorted(list(unique_target_fld_vals))
        bin_edges.append(bin_edges[-1] + 1)
        bin_counts = np.histogram(target_fld_vals, bin_edges)[0]
    else:
        bin_counts, bin_edges = np.histogram(target_fld_vals)

    # 3. grouping target_fld values by bin
    adjusted_bin_edges = get_adjusted_bin_edges(bin_edges)
    target_fld_bin_vals = get_target_fld_bin_vals(target_fld_vals, adjusted_bin_edges)

    # 4. adjusting bin_edges if any counts == 0
    has_zeros = any([not x for x in bin_counts])
    max_bin_size = 0.5 * len(df)
    has_huge_bin = any([x > max_bin_size for x in bin_counts])
    if has_zeros or has_huge_bin:
        if debug:
            print "## FIXING BAD BIN_EDGES ##"
            print "Edges before fix: "
            for bid, start in enumerate(adjusted_bin_edges[0:-1]):
                print " - ({:.0f}, {:.1f}): {}".format(start, adjusted_bin_edges[bid+1], bin_counts[bid])
            print "len(df) = {}, sum(bin_counts)={}\n".format(len(df), sum(bin_counts))

        target_fld_bin_vals, bin_counts, adjusted_bin_edges = \
                fix_problematic_bins(df, target_fld_vals, target_fld_bin_vals, bin_counts, adjusted_bin_edges, debug=debug)

        if debug:
            print "len(df) = {}, sum(bin_counts)={}\n".format(len(df), sum(bin_counts))

    return target_fld_bin_vals, bin_counts, adjusted_bin_edges


def fix_problematic_bins(df, target_fld_vals, target_fld_bin_vals, bin_counts, adjusted_bin_edges, debug=False):
    max_bin_size = 0.5 * len(df)
    tmp_bin_storage = {}  # stores (start, soft_end): [values in bin]

    for i in xrange(100):
        has_huge_bin = any([x > max_bin_size for x in bin_counts])
        if has_huge_bin:
            if debug:
                print "{}. Has big bin; counts={}".format(i, list(bin_counts))

            # 1. set aside existing bins
            max_bid = np.argmax(bin_counts)
            for bid in xrange(len(bin_counts)):
                if bid == max_bid:
                    continue
                start = adjusted_bin_edges[bid]
                end = adjusted_bin_edges[bid + 1]
                tmp_bin_storage[(start, end)] = target_fld_bin_vals[bid]

            # 2. split max bin to get new sub bins
            bin_counts, adjusted_bin_edges = np.histogram(target_fld_bin_vals[max_bid])
            target_fld_bin_vals = get_target_fld_bin_vals(target_fld_bin_vals[max_bid], adjusted_bin_edges)
        else:
            if debug:
                print "{}. No more huge bins: {}".format(i, list(bin_counts))
            for bid in xrange(len(bin_counts)):
                start = adjusted_bin_edges[bid]
                end = adjusted_bin_edges[bid + 1]
                tmp_bin_storage[(start, end)] = target_fld_bin_vals[bid]
            break
    if debug:
        print "Done with max bin resizing: len(tmp_bin_storage)={}".format(len(tmp_bin_storage))

    # building bin counts and edges
    bin_counts = []
    adjusted_bin_edges = []
    bin_edge_pairs = sorted(tmp_bin_storage.keys())
    for i, (start, end) in enumerate(bin_edge_pairs):
        adjusted_bin_edges.append(start)
        if i == len(bin_edge_pairs) - 1:
            adjusted_bin_edges.append(end)
        bin_counts.append(len(tmp_bin_storage[(start, end)]))

    # reducing counts and edges down to 10
    while len(bin_counts) > 10:
        min_bid = np.argmin(bin_counts)
        if min_bid == 0:
            del adjusted_bin_edges[1]
            bin_counts[0] += bin_counts[1]
            del bin_counts[1]
        else:
            del adjusted_bin_edges[min_bid]
            bin_counts[min_bid - 1] += bin_counts[min_bid]
            del bin_counts[min_bid]
    target_fld_bin_vals = get_target_fld_bin_vals(target_fld_vals, adjusted_bin_edges)
    return target_fld_bin_vals, bin_counts, adjusted_bin_edges


def get_target_fld_bin_vals(target_fld_vals, adjusted_bin_edges):
    target_fld_bin_vals = defaultdict(list)
    for val in target_fld_vals:
        for bid in xrange(len(adjusted_bin_edges) - 1):
            val_gte_start = (val >= adjusted_bin_edges[bid])
            val_lt_end = (val < adjusted_bin_edges[bid + 1])
            val_lte_end = (val <= adjusted_bin_edges[bid + 1])
            is_last = (bid == len(adjusted_bin_edges) - 2)
            if val_gte_start and (val_lt_end or (val_lte_end and is_last)):
                target_fld_bin_vals[bid].append(val)
                break
    return target_fld_bin_vals


def get_target_fld_vals_from_df(df, target_fld, random_seed=RANDOM_SEED):
    """
    generates binned values for target field
    """
    # 1. shuffling target_fld values
    np_random = np.random.RandomState(random_seed)
    target_fld_vals = list(df[target_fld])
    np_random.shuffle(target_fld_vals)

    # 2. getting bin_edges and bin_counts
    unique_target_fld_vals = df[target_fld].unique()
    if len(unique_target_fld_vals) <= 10:
        bin_edges = sorted(list(unique_target_fld_vals))
        bin_edges.append(bin_edges[-1] + 1)
        bin_counts = np.histogram(target_fld_vals, bin_edges)[0]
    else:
        bin_counts, bin_edges = np.histogram(target_fld_vals)

    # 3. grouping target_fld values by bin
    adjusted_bin_edges = get_adjusted_bin_edges(bin_edges)
    target_fld_bin_vals = get_target_fld_bin_vals(target_fld_vals, adjusted_bin_edges)

    # 4. adjusting bin_edges if any counts == 0
    has_zeros = any([not x for x in bin_counts])
    if has_zeros:
        max_bid = np.argmax(bin_counts)
        sub_bin_counts, sub_bin_edges = np.histogram(target_fld_bin_vals[max_bid])
        bin_counts = list(bin_counts)
        adjusted_bin_edges = list(adjusted_bin_edges)
        sub_bin_counts = list(sub_bin_counts)
        sub_bin_edges = list(sub_bin_edges)
        new_bin_counts = bin_counts[0:max_bid] + sub_bin_counts + bin_counts[max_bid+1:]
        new_edges = adjusted_bin_edges[0:max_bid] + sub_bin_edges[0:-1] + adjusted_bin_edges[max_bid+1:]
        while len(new_bin_counts) > 10:
            min_bid = np.argmin(new_bin_counts)
            if min_bid == 0:
                del new_edges[1]
                new_bin_counts[0] += new_bin_counts[1]
                del new_bin_counts[1]
            else:
                del new_edges[min_bid]
                new_bin_counts[min_bid - 1] += new_bin_counts[min_bid]
                del new_bin_counts[min_bid]
        bin_counts = new_bin_counts
        adjusted_bin_edges = new_edges
        target_fld_bin_vals = get_target_fld_bin_vals(target_fld_vals, adjusted_bin_edges)

    return target_fld_bin_vals, bin_counts, adjusted_bin_edges


def get_distrib_of_correlations(num_columns, percent_indep,
                                distrib="uniform", indep_slack=0.05,
                                percent_anti_dep=0.5, num_bins=10,
                                random_seed=RANDOM_SEED):
    """
    Returns list of dep scores for given params
    """
    np_random = np.random.RandomState(random_seed)

    num_corrs = int(round((num_columns**2 - num_columns) / 2.0))
    num_indep_corrs = int(round(num_corrs * percent_indep))
    num_dep_corrs = num_corrs - num_indep_corrs
    indep_corrs = []
    dep_corrs = []
    anti_dep_corrs = []

    # 1. if uniform, use uniform distrib to fill in desired dep strs
    if distrib == "uniform":
        # 1a. add independent scores
        while len(indep_corrs) < num_indep_corrs:
            score = np_random.uniform(0.0 - indep_slack, 0.0 + indep_slack + 0.000001)
            if score >= 0.0 - indep_slack and score <= 0.0 + indep_slack:
                indep_corrs.append(score)

        # 1b. add anti-dependent scores
        anti_dep_corrs += list(np_random.uniform(
            -1.0,
            0.0 - indep_slack,
            size=int(math.floor(num_dep_corrs * percent_anti_dep))
        ))

        # 1c. add dependent scores
        while len(dep_corrs) < int(math.ceil(num_dep_corrs * (1.0 - percent_anti_dep))):
            score = np_random.uniform(0.0 + indep_slack, 1.000001)
            if score > 0.0 + indep_slack and score <= 1.0:
                dep_corrs.append(score)

    # 2. if normal, use normal distrib to fill in desired dep strs
    elif distrib == "normal":
        # 2a. add independent scores
        while len(indep_corrs) < num_indep_corrs:
            scale = indep_slack / 4.0
            score = np_random.normal(loc=0.0, scale=scale)
            if score >= 0.0 - indep_slack and score <= 0.0 + indep_slack:
                indep_corrs.append(score)

        # 2b. add anti-dependent scores
        while len(anti_dep_corrs) < int(math.floor(num_dep_corrs * percent_anti_dep)):
            loc = (-1.0 - indep_slack) / 2.0
            scale = abs(loc / 4.0)
            score = np_random.normal(loc=loc, scale=scale)
            if score >= -1.0 and score < 0.0 - indep_slack:
                anti_dep_corrs.append(score)

        # 2c. add dependent scores
        while len(dep_corrs) < int(math.ceil(num_dep_corrs * (1.0 - percent_anti_dep))):
            loc = (1.0 + indep_slack) / 2.0
            scale = loc / 4.0
            score = np_random.normal(loc=loc, scale=scale)
            if score > 0.0 + indep_slack and score <= 1.0:
                dep_corrs.append(score)
    else:
        raise Exception("Invalid distribution specified.")

    # 3. return corrs
    corrs = anti_dep_corrs + indep_corrs + dep_corrs
    print "num_columns={}, num_corrs={}, num_dep_corrs={}, num_indep_corrs={}, len(scores)={}"\
        .format(num_columns, num_corrs, num_dep_corrs, num_indep_corrs, len(corrs))
    return corrs


def get_dep_str_bin_counts(scores, num_bins=10, indep_slack=0.05, adjusted_bin_edges=None):
    scores = np.array(scores)
    anti_dep_indexes = scores < 1.0 - indep_slack
    indep_indexes = (scores >= 1.0 - indep_slack) & (scores <= 1.0 + indep_slack)
    dep_indexes = scores > 1.0 + indep_slack
    anti_dep_scores = scores[anti_dep_indexes]
    indep_scores = scores[indep_indexes]
    dep_scores = scores[dep_indexes]

    if adjusted_bin_edges:
        scores = np.array(list(anti_dep_scores) + list(indep_scores) + list(dep_scores))
        bin_counts = []
        for i, bin_start in enumerate(adjusted_bin_edges[0:-1]):
            bin_end = adjusted_bin_edges[i + 1]
            bin_count = np.sum((scores >= bin_start) & (scores < bin_end))
            bin_counts.append(bin_count)
    else:
        anti_dep_num_bins = int(math.floor(num_bins / 2.0)) - 1
        anti_dep_bin_cnts, anti_dep_bin_edges = np.histogram(anti_dep_corrs, anti_dep_num_bins)
        adj_anti_dep_bin_edges = get_adjusted_bin_edges(anti_dep_bin_edges)
        dep_num_bins = int(math.ceil(num_bins / 2.0))
        dep_bin_cnts, dep_bin_edges = np.histogram(dep_corrs, dep_num_bins)
        adj_dep_bin_edges = get_adjusted_bin_edges(dep_bin_edges)
        adjusted_bin_edges = list(adj_anti_dep_bin_edges[:-1]) + [0.0 - indep_slack, 0.0 + indep_slack] + list(adj_dep_bin_edges[1:])
        bin_counts = list(anti_dep_bin_cnts) + [len(indep_corrs)] + list(dep_bin_cnts)

    return bin_counts, adjusted_bin_edges


def get_corr_bin_counts(corrs, num_bins=10, indep_slack=0.05, adjusted_bin_edges=None):
    corrs = np.array(corrs)
    anti_dep_indexes = corrs < 0.0 - indep_slack
    indep_indexes = (corrs >= 0.0 - indep_slack) & (corrs <= 0.0 + indep_slack)
    dep_indexes = corrs > 0.0 + indep_slack
    anti_dep_corrs = corrs[anti_dep_indexes]
    indep_corrs = corrs[indep_indexes]
    dep_corrs = corrs[dep_indexes]

    if adjusted_bin_edges:
        corrs = np.array(list(anti_dep_corrs) + list(indep_corrs) + list(dep_corrs))
        bin_counts = []
        for i, bin_start in enumerate(adjusted_bin_edges[0:-1]):
            bin_end = adjusted_bin_edges[i + 1]
            bin_count = np.sum((corrs >= bin_start) & (corrs < bin_end))
            bin_counts.append(bin_count)
    else:
        anti_dep_num_bins = int(math.floor(num_bins / 2.0)) - 1
        anti_dep_bin_cnts, anti_dep_bin_edges = np.histogram(anti_dep_corrs, anti_dep_num_bins)
        adj_anti_dep_bin_edges = get_adjusted_bin_edges(anti_dep_bin_edges)
        dep_num_bins = int(math.ceil(num_bins / 2.0))
        dep_bin_cnts, dep_bin_edges = np.histogram(dep_corrs, dep_num_bins)
        adj_dep_bin_edges = get_adjusted_bin_edges(dep_bin_edges)
        adjusted_bin_edges = list(adj_anti_dep_bin_edges[:-1]) + [0.0 - indep_slack, 0.0 + indep_slack] + list(adj_dep_bin_edges[1:])
        bin_counts = list(anti_dep_bin_cnts) + [len(indep_corrs)] + list(dep_bin_cnts)

    return bin_counts, adjusted_bin_edges


def get_corr_distrib(df, print_details=True):
    final_corrs = []
    col_pairs = itertools.combinations(df.columns, 2)
    for c1, c2 in col_pairs:
        final_corrs.append(pearsonr(df[c1], df[c2])[0])
    bin_counts, bin_edges = get_corr_bin_counts(final_corrs)
    adjusted_bin_edges = get_adjusted_bin_edges(bin_edges)
    if print_details:
        print "Column-pair correlation distrib: "
        for i, bin_start in enumerate(adjusted_bin_edges[0:-1]):
            print "- ({:.2f} - {:.2f}): {}".format(bin_start, adjusted_bin_edges[i+1], bin_counts[i])
    corr_avg = np.mean(final_corrs)
    corr_std = np.std(final_corrs)
    if print_details:
        print "avg: {:.3f}, stdev: {:.3f}, percent_indep: {:.3f}"\
            .format(corr_avg, corr_std, bin_counts[4] / float(len(final_corrs)))
        print("")
    return bin_counts, adjusted_bin_edges, corr_avg, corr_std


def print_bin_correlation_distrib(df, adjusted_bin_edges, target_fld):
    for bid, bin_start in enumerate(adjusted_bin_edges[0:-1]):
        bin_end = adjusted_bin_edges[bid + 1]
        qs = '({} >= {}) and ({} < {})'.format(target_fld, bin_start, target_fld, bin_end)
        bin_df = df.query(qs)
        if not len(bin_df):
            raise Exception("No counts for bid={}, qs={}".format(bid, qs))
        bin_corrs = []
        col_pairs = itertools.combinations(df.columns, 2)
        for c1, c2 in col_pairs:
            bin_corr = pearsonr(bin_df[c1], bin_df[c2])[0]
            bin_corrs.append(bin_corr)
        print "- bid={}, r={}, avg={:.2f}, stdev={:.2f}".format(bid, len(bin_df), np.mean(bin_corrs), np.std(bin_corrs))
    print("")











###############################################################################
############################ Data Gen Functions ###############################


def get_synthetic_data_v5(np_random, num_rows, num_columns, corr,
                          indep_slack=0.05, print_details=True):
    """
    Generates synthetic data form multivariate_normal distrib
    """
    # 1. create means, variances, covariance matrix
    num_corrs = int(round((num_columns**2 - num_columns) / 2.0))
    means = np.zeros(num_columns)
    variances = np.ones(num_columns)
    cov = squareform([corr] * num_corrs)
    for i in xrange(num_columns):
        cov[i, i] = 1.0

    # 2. generate data via multi-variate normal distribution
    data = np_random.multivariate_normal(means, cov, size=(num_rows,), check_valid='warn')
    cur_val = 0
    preds = []
    for cid in xrange(num_columns):
        data[:, cid] = np.where(data[:, cid] >= 0, cur_val, -1)
        preds.append(('c{}'.format(cid), '==', cur_val))
        cur_val += 1

    # 3. getting dataframe
    df = pd.DataFrame(data)
    df.columns = ['c{}'.format(i) for i in xrange(num_columns)]

    # 4. testing correlations
    if print_details:
        get_corr_distrib(df)

    return df, preds







###############################################################################
########################## Target Gen Functions ###############################


def get_target_fld_vals(num_rows, num_bins=10, target_answer_percent_uniform=0.333,
                        random_seed=RANDOM_SEED):
    """
    generates binned values for target field
    """
    np_random = np.random.RandomState(random_seed)

    target_fld_vals = list(np_random.normal(size=int(math.ceil(num_rows*(1-target_answer_percent_uniform)))))
    target_fld_vals += list(np_random.uniform(min(target_fld_vals), max(target_fld_vals), size=int(math.floor(num_rows*target_answer_percent_uniform))))
    np_random.shuffle(target_fld_vals)
    bin_counts, bin_edges = np.histogram(target_fld_vals, bins=num_bins)
    adjusted_bin_edges = get_adjusted_bin_edges(bin_edges)
    target_fld_bin_vals = defaultdict(list)
    for val in target_fld_vals:
        for bin_id in xrange(len(adjusted_bin_edges) - 1):
            if (val >= adjusted_bin_edges[bin_id] and
                    (val < adjusted_bin_edges[bin_id + 1] or
                     (bin_id == len(adjusted_bin_edges) - 2 and val <= adjusted_bin_edges[bin_id + 1]))):
                target_fld_bin_vals[bin_id].append(val)
                break
    return target_fld_bin_vals, bin_counts, adjusted_bin_edges


def add_target_to_df_v3(np_random, original_df, answer_preds, answer_corr, num_bins=10,
                        target_answer_percent_uniform=0.333, answer_slack=0.0,
                        target_fld='target_fld', print_details=True, min_meta=False):
    """
    Generates target answer in df
    """
    df = copy.deepcopy(original_df)
    num_rows = len(df)
    meta = {}

    # 0.5 more meta
    meta['pre_gen_pred_pair_corr_distrib_counts'], \
        meta['pre_gen_pred_pair_corr_distrib_edges'], \
        meta['pre_gen_pred_pair_corr_distrib_avg'], \
        meta['pre_gen_pred_pair_corr_distrib_std'] = get_corr_distrib(df, print_details=False)

    # 1. getting target values and bins; outputting correlations for answer preds
    target_fld_bin_vals, bin_counts, adjusted_bin_edges = get_target_fld_vals(num_rows)
    p1, p2 = answer_preds
    bin_corrs = []
    meta['pre_gen_answer_col_corr'] = pearsonr(df[p1[0]], df[p2[0]])[0]
    if print_details:
        print("Pre t-gen v3 answer col corr={:.3f}".format(meta['pre_gen_answer_col_corr']))
        print("Pre t-gen v3 per-bin answer-col correlations:")
    meta['pre_gen_answer_bin_corrs'] = []
    meta['pre_gen_answer_bin_dep_strs'] = []
    for bid, r in enumerate(bin_counts):
        start = sum(bin_counts[0:bid])
        end = sum(bin_counts[0:bid + 1])
        bin_df = df.loc[start:end - 1, :]
        cc = len(bin_df.query(Query.get_pandas_query_from_preds(answer_preds)))
        c1 = len(bin_df.query(Query.get_pandas_query_from_preds([p1])))
        c2 = len(bin_df.query(Query.get_pandas_query_from_preds([p2])))
        bin_corr, bin_pval = pearsonr(bin_df[p1[0]], bin_df[p2[0]])
        dep_str = (r * cc) / float(c1 * c2)
        meta['pre_gen_answer_bin_corrs'].append(bin_corr)
        meta['pre_gen_answer_bin_dep_strs'].append(dep_str)
        if print_details:
            print "- bid: {}, dep_str={:.3f}, corr(c0, c1): {:.3f} ({:.2f})".format(bid, dep_str, bin_corr, bin_pval)
        bin_corrs.append(bin_corr)
    if print_details:
        print "avg: {:.2f}, std: {:.2f}".format(np.mean(np.abs(bin_corrs)), np.std(np.abs(bin_corrs)))
        print("")

    # 1.5. DEBUG:
    if not min_meta:
        meta['pre_gen_pred_pair_avg_bin_corrs'] = []
        meta['pre_gen_pred_pair_avg_bin_stds'] = []
        if print_details:
            print("Pre t-gen v3 per-bin pred-pair correlations:")
        for bid, r in enumerate(bin_counts):
            start = sum(bin_counts[0:bid])
            end = sum(bin_counts[0:bid + 1])
            bin_df = df.loc[start:end - 1, :]
            if not len(bin_df):
                raise Exception("No counts for bid={}, qs={}".format(bid, qs))
            bin_corrs = []
            col_pairs = itertools.combinations(df.columns, 2)
            for c1, c2 in col_pairs:
                bin_corr = pearsonr(bin_df[c1], bin_df[c2])[0]
                bin_corrs.append(bin_corr)
            meta['pre_gen_pred_pair_avg_bin_corrs'].append(np.mean(bin_corrs))
            meta['pre_gen_pred_pair_avg_bin_stds'].append(np.std(bin_corrs))
            if print_details:
                print "- bid={}, r={}, avg={:.2f}, stdev={:.2f}".format(bid, len(bin_df), np.mean(bin_corrs), np.std(bin_corrs))
        meta['pre_gen_pred_pair_bin_corrs_avg'] = np.mean(meta['pre_gen_pred_pair_avg_bin_corrs'])
        meta['pre_gen_pred_pair_bin_corrs_std'] = np.std(meta['pre_gen_pred_pair_avg_bin_corrs'])
        if print_details:
            print("avg: {:.2f}, std: {:.2f}".format(meta['pre_gen_pred_pair_bin_corrs_avg'], meta['pre_gen_pred_pair_bin_corrs_std']))
            print("")

    # 2. determining average overlap (as initial new_cc)
    cnt_data = []
    for bid, r in enumerate(bin_counts):
        start = sum(bin_counts[0:bid])
        end = sum(bin_counts[0:bid + 1])
        bin_df = df.loc[start:end - 1, :]
        cc = len(bin_df.query(Query.get_pandas_query_from_preds(answer_preds)))
        c1 = len(bin_df.query(Query.get_pandas_query_from_preds([p1])))
        c2 = len(bin_df.query(Query.get_pandas_query_from_preds([p2])))
        cnt_data.append((r, cc, c1, c2, start, end))
    new_cc = int(round(np.mean([x[1] for x in cnt_data])))
    meta['pre_gen_answer_bin_counts'] = [x[1] for x in cnt_data]
    meta['initial_new_cc'] = new_cc
    if print_details:
        print "Pre t-gen v3 answer counts per bin: {} (avg={})".format([x[1] for x in cnt_data], new_cc)

    # 3. finding new overlap (must allow for enough rows to overlap w/ desired probability)
    while True:
        bad_cc = False
        for bid, (r, cc, c1, c2, start, end) in enumerate(cnt_data):
            dep_str = 1.0 + answer_corr  # old way: (r * cc) / float(c1 * c2)
            new_c = int(round(np.sqrt((r * new_cc) / dep_str)))
            if ((2 * (new_c - new_cc)) + new_cc) > r:
                bad_cc = True
                break
            if new_c > r or new_cc > new_c:
                bad_cc = True
                break
        if not bad_cc:
            break
        else:
            new_cc -= 1
        if new_cc <= 0:
            raise Exception("new_cc = {}. exiting".format(new_cc))
    meta['final_new_cc'] = new_cc
    meta['final_new_c'] = new_c

    # 4. adjust c1 and c2 to minimize correlation change from changing cc to avg_cc
    target_counts = []
    for bid, (r, cc, c1, c2, start, end) in enumerate(cnt_data):
        # 4a. determining new_c value
        dep_str = 1.0 + answer_corr  # old way: (r * cc) / float(c1 * c2)
        new_c = int(round(np.sqrt((r * new_cc) / dep_str)))

        # 4b. getting indexes for val combos of c1=v1, c1=-1, c2=v2, c2=-1
        bin_df = df.loc[start:end - 1, :]
        val_combo_indexes = {
            'nn': list(bin_df.query('({} == -1) and ({} == -1)'.format(p1[0], p2[0])).index.values),
            'vn': list(bin_df.query('({} == {}) and ({} == -1)'.format(p1[0], p1[2], p2[0])).index.values),
            'nv': list(bin_df.query('({} == -1) and ({} == {})'.format(p1[0], p2[0], p2[2])).index.values),
            'vv': list(bin_df.query('({} == {}) and ({} == {})'.format(p1[0], p1[2], p2[0], p2[2])).index.values),
        }
        for key in val_combo_indexes.keys():
            np_random.shuffle(val_combo_indexes[key])
        # print "=== bid={}, r={}, cc={}, new_cc={}, c1={}, c2={}, new_c={} ===".format(bid, r, cc, new_cc, c1, c2, new_c)
        # print " - val_combo_indexes counts: {}".format(', '.join(['{}={}'.format(k, len(v)) for k, v in val_combo_indexes.iteritems()]))

        # 4c. getting +/- counts from each combo's relation w/ new_c & new_cc
        change_counts = {
            'vv': new_cc - len(val_combo_indexes['vv']),
            'nn': (r - new_cc - (2 * (new_c - new_cc))) - len(val_combo_indexes['nn']),
            'vn': (new_c - new_cc) - len(val_combo_indexes['vn']),
            'nv': (new_c - new_cc) - len(val_combo_indexes['nv']),
        }
        # print " - change counts: {}".format(', '.join(['{}={}'.format(k, v) for k, v in change_counts.iteritems()]))

        # 4d. determining add and remove indexes
        add_cnt = 0
        rm_cnt = 0
        add_key_counts = {}
        alter_indexes = {}
        for key, cnt in change_counts.iteritems():
            if cnt > 0:
                add_cnt += cnt
                add_key_counts[key] = cnt
            else:
                rm_cnt += len(val_combo_indexes[key][0:abs(cnt)])
                alter_indexes[key] = val_combo_indexes[key][0:abs(cnt)]
        # print " - add_cnt={}, rm_cnt={}".format(add_cnt, rm_cnt)
        if add_cnt != rm_cnt:
            raise Exception("Error: add_cnt={} != rm_cnt={}".format(add_cnt, rm_cnt))

        # 4e. moving indexes around to match final counts
        add_keys = add_key_counts.keys()
        alter_keys = alter_indexes.keys()
        add_key_index = 0
        alter_key_index = 0
        for i in xrange(add_cnt):
            # find the next valid add_key
            has_valid_add_key = False
            for j in xrange(len(add_keys)):
                add_key = add_keys[add_key_index % len(add_keys)]
                add_key_index += 1
                if add_key_counts[add_key] > 0:
                    has_valid_add_key = True
                    add_key_counts[add_key] -= 1
                    break
            if not has_valid_add_key:
                raise Exception("Error: no valid add_key found!")

            # find the next valid alter_key
            has_valid_alter_key = False
            for j in xrange(len(alter_keys)):
                alter_key = alter_keys[alter_key_index % len(alter_keys)]
                alter_key_index += 1
                if len(alter_indexes[alter_key]) > 0:
                    has_valid_alter_key = True
                    alter_index = alter_indexes[alter_key].pop()
                    break
            if not has_valid_alter_key:
                raise Exception("Error: no valid alter_key found!")

            # switching alter_index's config to add_key config
            if add_key == 'nn' and alter_key in ('vn', 'nv', 'vv'):
                df.loc[alter_index, [p1[0], p2[0]]] = [-1, -1]
            elif add_key == 'vn' and alter_key in ('nn', 'nv', 'vv'):
                df.loc[alter_index, [p1[0], p2[0]]] = [p1[2], -1]
            elif add_key == 'nv' and alter_key in ('nn', 'vn', 'vv'):
                df.loc[alter_index, [p1[0], p2[0]]] = [-1, p2[2]]
            elif add_key == 'vv' and alter_key in ('nn', 'vn', 'nv'):
                df.loc[alter_index, [p1[0], p2[0]]] = [p1[2], p2[2]]
            else:
                raise Exception("Invalid add_key / alter_key combo")
        # print " - final add_key_counts = {}".format(', '.join(['{}={}'.format(k, v) for k, v in add_key_counts.iteritems()]))
        # print " - final alter_indexes counts = {}".format(', '.join(['{}={}'.format(k, len(v)) for k, v in alter_indexes.iteritems()]))

        # 4f. updating target_counts
        bin_df = df.loc[start:end - 1, :]
        actual_cc = len(bin_df.query(Query.get_pandas_query_from_preds(answer_preds)))
        target_counts.append(actual_cc)

    # 5. if answer_slack > 0, update target counts to have +/- some slack
    answer_target_counts = copy.deepcopy(target_counts)
    if answer_slack > 0.0:
        for bid, targ_cnt in enumerate(target_counts):
            slack_sd = answer_slack * targ_cnt * 0.25
            cnt = int(round(np_random.normal(loc=targ_cnt, scale=slack_sd, size=1)[0]))
            if cnt > targ_cnt + (answer_slack * targ_cnt):
                cnt = targ_cnt + (answer_slack * targ_cnt)
            elif cnt < targ_cnt - (answer_slack * targ_cnt):
                cnt = targ_cnt - (answer_slack * targ_cnt)
            answer_target_counts[bid] = cnt
    # print "answer_target_counts: {} (sum={})".format(answer_target_counts, sum(answer_target_counts))

    # 6. adding target answer column to dataframe
    final_target_fld_vals = np.zeros((num_rows, 1))
    answer_df = df.query(Query.get_pandas_query_from_preds(answer_preds))
    answer_indexes = list(answer_df.index.values)
    # print "len(answer_indexes)={}".format(len(answer_indexes))
    remaining_indexes = list(set(range(0, num_rows)) - set(answer_indexes))
    np_random.shuffle(answer_indexes)
    remaining_target_fld_vals = []
    for bid, cnt in enumerate(answer_target_counts):
        row_ids = answer_indexes[0:cnt]
        # print("row_ids={}, t1={}, t2={}".format(len(row_ids), len(target_fld_bin_vals[bid][0:cnt]), len(target_fld_bin_vals[bid][cnt:])))
        del answer_indexes[0:cnt]
        final_target_fld_vals[row_ids, 0] = target_fld_bin_vals[bid][0:cnt]
        remaining_target_fld_vals += target_fld_bin_vals[bid][cnt:]
    final_target_fld_vals[remaining_indexes, 0] = remaining_target_fld_vals
    df[target_fld] = final_target_fld_vals

    # 7. debugging
    meta['post_gen_answer_col_corr'] = pearsonr(df[p1[0]], df[p2[0]])[0]
    if print_details:
        print "Post t-gen v3 answer col corr={:.3f}".format(meta['post_gen_answer_col_corr'])
        print "Post t-gen v3 per-bin correlations:"
    meta['post_gen_answer_bin_corrs'] = []
    meta['post_gen_answer_bin_dep_strs'] = []
    bin_corrs = []
    for bid, bin_start in enumerate(adjusted_bin_edges[0:-1]):
        bin_end = adjusted_bin_edges[bid + 1]
        bin_df = df.query("({} >= {}) and ({} < {})".format(target_fld, bin_start, target_fld, bin_end))
        bin_corr = pearsonr(bin_df[p1[0]], bin_df[p2[0]])[0]
        bin_corrs.append(bin_corr)
        cc = len(bin_df.query(Query.get_pandas_query_from_preds(answer_preds)))
        c1 = len(bin_df.query(Query.get_pandas_query_from_preds([p1])))
        c2 = len(bin_df.query(Query.get_pandas_query_from_preds([p2])))
        dep_str = (r * cc) / float(c1 * c2)
        meta['post_gen_answer_bin_corrs'].append(bin_corr)
        meta['post_gen_answer_bin_dep_strs'].append(dep_str)
        if print_details:
            print "- bid={}, corr: {:.3f}".format(bid, bin_corr)
    if print_details:
        print "avg: {:.2f}, std: {:.2f}".format(np.mean(np.abs(bin_corrs)), np.std(np.abs(bin_corrs)))
    if not min_meta:
        if print_details:
            print("\nPost t-gen v3 per-bin pred-pair correlations:")
        meta['post_gen_pred_pair_avg_bin_corrs'] = []
        meta['post_gen_pred_pair_avg_bin_stds'] = []
        for bid, bin_start in enumerate(adjusted_bin_edges[0:-1]):
            bin_end = adjusted_bin_edges[bid + 1]
            bin_df = df.query("({} >= {}) and ({} < {})".format(target_fld, bin_start, target_fld, bin_end))
            bin_corrs = []
            col_pairs = itertools.combinations(df.columns, 2)
            for c1, c2 in col_pairs:
                bin_corr = pearsonr(bin_df[c1], bin_df[c2])[0]
                bin_corrs.append(bin_corr)
            meta['post_gen_pred_pair_avg_bin_corrs'].append(np.mean(bin_corrs))
            meta['post_gen_pred_pair_avg_bin_stds'].append(np.std(bin_corrs))
            if print_details:
                print "- bid={}, r={}, avg={:.2f}, stdev={:.2f}".format(bid, len(bin_df), np.mean(bin_corrs), np.std(bin_corrs))
        meta['post_gen_pred_pair_bin_corrs_avg'] = np.mean(meta['post_gen_pred_pair_avg_bin_corrs'])
        meta['post_gen_pred_pair_bin_corrs_std'] = np.std(meta['post_gen_pred_pair_avg_bin_corrs'])
        if print_details:
            print("avg: {:.2f}, std: {:.2f}".format(meta['post_gen_pred_pair_bin_corrs_avg'], meta['post_gen_pred_pair_bin_corrs_std']))
            print("")

    # 7. return results (reminder: df edited inline)
    meta['target_fld'] = target_fld
    meta['target_counts'] = target_counts
    meta['adjusted_bin_edges'] = adjusted_bin_edges
    meta['bin_counts'] = np.histogram(df[target_fld], adjusted_bin_edges)[0]
    return df, meta


def get_synthetic_data_and_answer_v1(num_rows, num_columns, num_preds, corr, random_seed,
                                     data_cache_file, print_details=False,
                                     min_meta=False):
    """
    Creates df, preds, synthetic answer
    """
    if os.path.exists(data_cache_file):
        with open(data_cache_file) as f:
            df, preds, meta = pickle.load(f)
    else:
        start = timer()
        np_random = np.random.RandomState(random_seed)

        # 1. getting synthetic data
        df, preds = get_synthetic_data_v5(np_random, num_rows, num_columns, corr,
                                          print_details=print_details)

        # 2. randomly choosing answer preds
        answer_preds = []
        answer_pids = []
        for i in xrange(100):
            pid1 = np_random.randint(0, len(preds))
            pid2 = np_random.randint(0, len(preds))
            if pid1 != pid2:
                answer_preds = [preds[pid1], preds[pid2]]
                answer_pids = sorted([pid1, pid2])
                break
        if not answer_preds:
            raise Exception("Problem getting unique answer preds.. 100 iterations matched.")
        if print_details:
            print "answer preds: {}".format(answer_preds)
            print "answer pids: {}".format(answer_pids)

        # 3. adding synthetic answer to df
        df, meta = add_target_to_df_v3(np_random, df, answer_preds, corr,
                                       print_details=print_details, min_meta=min_meta)

        # 4. noting some meta data
        meta['answer_preds'] = answer_preds
        meta['answer_pids'] = answer_pids
        meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)
        qs = Query.get_pandas_query_from_preds(answer_preds)
        meta['answer_counts'] = np.histogram(df.query(qs)[meta['target_fld']], meta['adjusted_bin_edges'])[0]

        # 5. caching results
        meta['data_gen_runtime'] = timer() - start
        with open(data_cache_file, "w") as f:
            pickle.dump((df, preds, meta), f, -1)

    return df, preds, meta


def get_synthetic_data_and_answer_v9(num_rows, num_columns, num_preds, corr, random_seed,
                                     data_cache_file, print_details=False,
                                     min_meta=False, ignore_data_cache=False,
                                     target_fld='target_fld',

                                     # this is the percent of rows outside of answer
                                     # tuples to throw off individ corr
                                     percent_non_answer_used=0.7,

                                     # what percent of the above percent goes to
                                     # answer1; remainder to answer2
                                     ans1_percent_non_answer_used=0.5,

                                     # percent of answer and non-answer used by
                                     # noise column
                                     noise_percent_answer_used=0.7,
                                     noise_percent_non_answer_used=0.1):
    """
    Creates df, preds, synthetic answer
    """
    if not ignore_data_cache and os.path.exists(data_cache_file):
        with open(data_cache_file) as f:
            df, preds, meta = pickle.load(f)
    else:
        if ignore_data_cache:
            print "WARNING: data cache being ignored!"

        start = timer()
        np_random = np.random.RandomState(random_seed)
        meta = {}

        # 1. create means, variances, covariance matrix
        num_corrs = int(round((num_columns**2 - num_columns) / 2.0))
        means = np.zeros(num_columns)
        cov = squareform([corr] * num_corrs)
        for i in xrange(num_columns):
            cov[i, i] = 1.0

        # 2. creating dataset via normal distrib
        data = np_random.multivariate_normal(means, cov, size=(num_rows,), check_valid='warn')
        df = pd.DataFrame(data)
        df.columns = ['c{}'.format(i) for i in xrange(num_columns)]
        if print_details:
            get_corr_distrib(df)

        # 3. getting synthetic answer and meta
        num_rows = len(df)
        target_fld_bin_vals, bin_counts, adjusted_bin_edges = get_target_fld_vals(num_rows)
        target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, ans_preds, noise_preds = \
                get_synthetic_answer_v2(df, target_fld_bin_vals, bin_counts,
                                        percent_non_answer_used=percent_non_answer_used,
                                        ans1_percent_non_answer_used=ans1_percent_non_answer_used,
                                        noise_percent_answer_used=noise_percent_answer_used,
                                        noise_percent_non_answer_used=noise_percent_non_answer_used,
                                        print_details=print_details)
        c_preds = ans_preds + noise_preds

        # 6. add target/answer/noise fields to dataframe
        df.columns = [target_fld, 'answer1', 'answer2', 'noise1'] + list(df.columns[4:])
        df[target_fld] = target_fld_vals
        df['answer1'] = answer1_vals
        df['answer2'] = answer2_vals
        df['noise1'] = noise1_vals

        # 7. print correlations
        if print_details:
            ans1_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[0]]))['target_fld'])[0]
            print "ans1 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans1_signal, pearsonr(target_counts, ans1_signal)[0], get_signal_distance(target_counts, ans1_signal))

            ans2_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[1]]))['target_fld'])[0]
            print "ans2 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans2_signal, pearsonr(target_counts, ans2_signal)[0], get_signal_distance(target_counts, ans2_signal))

            ans_signal = np.histogram(df.query(Query.get_pandas_query_from_preds(c_preds[0:2]))['target_fld'])[0]
            print "ans signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans_signal, pearsonr(target_counts, ans_signal)[0], get_signal_distance(target_counts, ans_signal))

            print('---------------------')

            noise1_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[2]]))['target_fld'])[0]
            print "noise1 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(noise1_signal, pearsonr(target_counts, noise1_signal)[0], get_signal_distance(target_counts, noise1_signal))

            ans1_noise1_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[0], c_preds[2]]))['target_fld'])[0]
            print "ans1 noise1 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans1_noise1_signal, pearsonr(target_counts, ans1_noise1_signal)[0], get_signal_distance(target_counts, ans1_noise1_signal))

            ans2_noise1_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[1], c_preds[2]]))['target_fld'])[0]
            print "ans2 noise1 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans2_noise1_signal, pearsonr(target_counts, ans2_noise1_signal)[0], get_signal_distance(target_counts, ans2_noise1_signal))

            print("Avg column corr:")
            print_avg_column_corr(df)

        # 8. generate predicates
        # generate predicates {eq, gt, lt} for all unique columns+values; take random subset
        rand_preds = get_rand_predicates(np_random, df, c_preds, target_fld, num_preds-len(c_preds))
        preds = c_preds + rand_preds
        if print_details:
            print("Num preds: {}".format(len(preds)))
            print preds[0:10]

        # 9. setting meta and caching results
        meta['target_fld'] = target_fld
        meta['target_counts'] = target_counts
        meta['adjusted_bin_edges'] = adjusted_bin_edges
        meta['bin_counts'] = bin_counts
        meta['answer_preds'] = preds[0:2]
        meta['answer_pids'] = [0, 1]
        meta['answer_query'] = Query.get_pandas_query_from_preds(meta['answer_preds'])
        meta['answer_counts'] = np.histogram(df.query(meta['answer_query'])[meta['target_fld']], meta['adjusted_bin_edges'])[0]
        meta['data_gen_runtime'] = timer() - start

        # 9. caching results
        meta['data_gen_runtime'] = timer() - start
        with open(data_cache_file, "w") as f:
            pickle.dump((df, preds, meta), f, -1)

    if print_details:
        print("Done building syn data v9: {} sec".format(meta['data_gen_runtime']))
    return df, preds, meta


def get_rand_predicates(np_random, df, c_preds, target_fld, num_preds):
    all_preds = []
    for col in df.columns:
        if col == target_fld:
            continue
        unique_vals = df[col].unique()
        for unique_val in unique_vals:
            eq_pred = (col, '==', unique_val)
            if eq_pred not in c_preds:
                all_preds.append(eq_pred)
            lt_pred = (col, '<', unique_val)
            if lt_pred not in c_preds:
                all_preds.append(lt_pred)
            gt_pred = (col, '>', unique_val)
            if gt_pred not in c_preds:
                all_preds.append(gt_pred)
    np_random.shuffle(all_preds)
    return all_preds[0:num_preds]


def get_synthetic_data_and_answer_v2(num_rows, num_columns, num_preds, random_seed,
                                     data_cache_file, target_fld='target_fld',
                                     print_details=False,

                                     # min value for columns' values
                                     min_field_val=0,

                                     # max value for columns' values
                                     max_field_val=100,

                                     # this is the percent of rows outside of answer
                                     # tuples to throw off individ corr
                                     percent_non_answer_used=0.7,

                                     # what percent of the above percent goes to
                                     # answer1; remainder to answer2
                                     ans1_percent_non_answer_used=0.5,

                                     # percent of answer and non-answer used by
                                     # noise column
                                     noise_percent_answer_used=0.7,
                                     noise_percent_non_answer_used=0.1):
    """
    Creates df, preds, synthetic answer in v2 manner
    """
    if print_details:
        print("## Building / Getting Syn Data V2 ##")

    meta = {}
    if os.path.exists(data_cache_file):
        if print_details:
            print("Loading syn data v2 from cache")
        with open(data_cache_file) as f:
            df, preds, meta = pickle.load(f)
    else:
        start = timer()

        # 1. randomly fill matrix uniformly
        np_random = np.random.RandomState(random_seed)
        data = np_random.randint(min_field_val, max_field_val, [num_rows, num_columns])
        df = pd.DataFrame(data)
        df.columns = ['c{}'.format(i) for i in xrange(num_columns)]

        # 2 getting synthetic answer and meta
        target_fld_bin_vals, bin_counts, adjusted_bin_edges = get_target_fld_vals(num_rows)
        target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, ans_preds, noise_preds = \
                get_synthetic_answer_v2(df, target_fld_bin_vals, bin_counts,
                                        percent_non_answer_used=percent_non_answer_used,
                                        ans1_percent_non_answer_used=ans1_percent_non_answer_used,
                                        noise_percent_answer_used=noise_percent_answer_used,
                                        noise_percent_non_answer_used=noise_percent_non_answer_used,
                                        print_details=print_details)
        c_preds = ans_preds + noise_preds

        # 6. add target/answer/noise fields to dataframe
        df.columns = [target_fld, 'answer1', 'answer2', 'noise1'] + list(df.columns[4:])
        df[target_fld] = target_fld_vals
        df['answer1'] = answer1_vals
        df['answer2'] = answer2_vals
        df['noise1'] = noise1_vals

        # 7. print correlations
        if print_details:
            ans1_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[0]]))['target_fld'])[0]
            print "ans1 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans1_signal, pearsonr(target_counts, ans1_signal)[0], get_signal_distance(target_counts, ans1_signal))

            ans2_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[1]]))['target_fld'])[0]
            print "ans2 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans2_signal, pearsonr(target_counts, ans2_signal)[0], get_signal_distance(target_counts, ans2_signal))

            ans_signal = np.histogram(df.query(Query.get_pandas_query_from_preds(c_preds[0:2]))['target_fld'])[0]
            print "ans signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans_signal, pearsonr(target_counts, ans_signal)[0], get_signal_distance(target_counts, ans_signal))

            print('---------------------')

            noise1_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[2]]))['target_fld'])[0]
            print "noise1 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(noise1_signal, pearsonr(target_counts, noise1_signal)[0], get_signal_distance(target_counts, noise1_signal))

            ans1_noise1_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[0], c_preds[2]]))['target_fld'])[0]
            print "ans1 noise1 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans1_noise1_signal, pearsonr(target_counts, ans1_noise1_signal)[0], get_signal_distance(target_counts, ans1_noise1_signal))

            ans2_noise1_signal = np.histogram(df.query(Query.get_pandas_query_from_preds([c_preds[1], c_preds[2]]))['target_fld'])[0]
            print "ans2 noise1 signal: {}, corr={:.3f}, sig_dist={:.3f}"\
                .format(ans2_noise1_signal, pearsonr(target_counts, ans2_noise1_signal)[0], get_signal_distance(target_counts, ans2_noise1_signal))

            print("Avg column corr:")
            print_avg_column_corr(df)

        # 8. generate predicates
        # generate predicates {eq, gt, lt} for all unique columns+values; take random subset
        rand_preds = get_rand_predicates(np_random, df, c_preds, target_fld, num_preds-len(c_preds))
        preds = c_preds + rand_preds
        if print_details:
            print("Num preds: {}".format(len(preds)))
            print preds[0:10]

        # 9. setting meta and caching results
        meta['target_fld'] = target_fld
        meta['target_counts'] = target_counts
        meta['adjusted_bin_edges'] = adjusted_bin_edges
        meta['bin_counts'] = bin_counts
        meta['answer_preds'] = preds[0:2]
        meta['answer_pids'] = (0, 1)
        meta['answer_query'] = Query.get_pandas_query_from_preds(meta['answer_preds'])
        meta['answer_counts'] = np.histogram(df.query(meta['answer_query'])[meta['target_fld']], meta['adjusted_bin_edges'])[0]
        meta['data_gen_runtime'] = timer() - start
        with open(data_cache_file, "w") as f:
            pickle.dump((df, preds, meta), f, -1)

        if print_details:
            print("Done building syn data v2: {} sec".format(meta['data_gen_runtime']))

    return df, preds, meta


def get_synthetic_answer_v2(df, target_fld_bin_vals, bin_counts, print_details=False,

                            # this is the percent of rows outside of answer
                            # tuples to throw off individ corr
                            percent_non_answer_used=0.7,

                            # what percent of the above percent goes to
                            # answer1; remainder to answer2
                            ans1_percent_non_answer_used=0.5,

                            # percent of answer and non-answer used by
                            # noise column
                            noise_percent_answer_used=0.7,
                            noise_percent_non_answer_used=0.1):
    """
    builds synthetic answer columns
    """
    # 2. building target column values and target signal
    target_counts = np.zeros(len(bin_counts), dtype=int)
    for index in (0, 1, -2, -1):
        target_counts[index] = int(round(0.9 * bin_counts[index]))
    if print_details:
        print "bin counts: {}".format(bin_counts)
        print "target signal: {}".format(target_counts)

    # 3. creating synthetic predicates 1 & 2, each poorly correlated w/ target until combine
    target_fld_vals = []
    answer1_indexes = []
    answer2_indexes = []
    noise1_indexes = []
    for bid in xrange(len(bin_counts)):
        # 3a. note which of the target values belong to an answer
        cur_answer_vals = target_fld_bin_vals[bid][0:target_counts[bid]]
        cur_answer_indexes = range(len(target_fld_vals), len(target_fld_vals) + len(cur_answer_vals))
        answer1_indexes += cur_answer_indexes
        answer2_indexes += cur_answer_indexes

        # 3b. inverting some of non-answer indexes b/w ans1 and ans2 to throw off individ corr
        num_non_answers_used = int(round(percent_non_answer_used * len(target_fld_bin_vals[bid][target_counts[bid]:])))
        ans1_num_non_answers_used = int(math.ceil(ans1_percent_non_answer_used * num_non_answers_used))
        ans2_num_non_answers_used = int(math.floor(ans1_percent_non_answer_used * num_non_answers_used))
        ans1_noise_start = len(target_fld_vals) + target_counts[bid]
        ans1_noise_end = ans1_noise_start + ans1_num_non_answers_used
        ans2_noise_end = ans1_noise_end + ans2_num_non_answers_used
        cur_ans1_noise_vals = target_fld_bin_vals[bid][ans1_noise_start:ans1_noise_end]
        cur_ans2_noise_vals = target_fld_bin_vals[bid][ans1_noise_end:ans2_noise_end]
        answer1_indexes += range(len(target_fld_vals) + ans1_noise_start, len(target_fld_vals) + ans1_noise_end)
        answer2_indexes += range(len(target_fld_vals) + ans1_noise_end, len(target_fld_vals) + ans2_noise_end)

        # 3c. choosing noise1 indexes
        noise1_num_answers_used = int(round(noise_percent_answer_used * len(target_fld_bin_vals[bid][0:target_counts[bid]])))
        noise1_num_non_answers_used = int(round(noise_percent_non_answer_used * len(target_fld_bin_vals[bid][target_counts[bid]:])))
        noise1_indexes += range(len(target_fld_vals), len(target_fld_vals) + noise1_num_answers_used)
        noise1_indexes += range(len(target_fld_vals) + len(target_fld_bin_vals[bid]),
                                len(target_fld_vals) + len(target_fld_bin_vals[bid]) + noise1_num_non_answers_used)

        # 3d. add current bin's target vals to final target vals signal
        target_fld_vals += target_fld_bin_vals[bid]

    # 4. fill in synthetic answer's values
    cur_ans1_val = 0
    cur_ans2_val = 0
    answer1_vals = []
    answer2_vals = []
    answer1_indexes = set(answer1_indexes)
    answer2_indexes = set(answer2_indexes)
    for i in xrange(len(df)):
        if i in answer1_indexes:
            ans1_val = cur_ans1_val
            cur_ans1_val += 1
        else:
            ans1_val = 99999999
        answer1_vals.append(ans1_val)

        if i in answer2_indexes:
            ans2_val = cur_ans2_val
            cur_ans2_val += 1
        else:
            ans2_val = 99999999
        answer2_vals.append(ans2_val)

    # 5. create noise column
    cur_noise1_val = 0
    noise1_vals = []
    noise1_indexes = set(noise1_indexes)
    for i in xrange(len(df)):
        if i in noise1_indexes:
            noise1_val = cur_noise1_val
            cur_noise1_val += 1
        else:
            noise1_val = 99999999
        noise1_vals.append(noise1_val)

    # 6. adding synthetic answer's preds
    ans_preds = [
        ('answer1', '<', cur_ans1_val),
        ('answer2', '<', cur_ans2_val),
    ]
    noise_preds = [
        ('noise1', '<', cur_noise1_val),
    ]
    return target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, ans_preds, noise_preds


def get_synthetic_answer_v3(df, target_fld_bin_vals, bin_counts, print_details=False,

                            # this is the percent of rows outside of answer
                            # tuples to throw off individ corr
                            percent_non_answer_used=0.7,

                            # what percent of the above percent goes to
                            # answer1; remainder to answer2
                            ans1_percent_non_answer_used=0.5,

                            # percent of answer and non-answer used by
                            # noise column
                            noise_percent_answer_used=0.7,
                            noise_percent_non_answer_used=0.1):
    """
    builds synthetic answer columns
    """
    # 2. building target column values and target signal
    target_counts = np.zeros(len(bin_counts), dtype=int)
    for index in (0, 1, -2, -1):
        target_counts[index] = int(round(0.9 * bin_counts[index]))
    if print_details:
        print "bin counts: {}".format(bin_counts)
        print "target signal: {}".format(target_counts)

    # 3. creating synthetic predicates 1 & 2, each poorly correlated w/ target until combine
    target_fld_vals = []
    answer1_indexes = []
    answer2_indexes = []
    noise1_indexes = []
    for bid in xrange(len(bin_counts)):
        # 3a. note which of the target values belong to an answer
        cur_answer_indexes = range(len(target_fld_vals), len(target_fld_vals) + target_counts[bid])
        answer1_indexes += cur_answer_indexes
        answer2_indexes += cur_answer_indexes

        ## 3b. inverting some of non-answer indexes b/w ans1 and ans2 to throw off individ corr ##
        num_non_answers = int(round(percent_non_answer_used * len(target_fld_bin_vals[bid][target_counts[bid]:])))
        na_indexes = range(len(target_fld_vals) + target_counts[bid], len(target_fld_vals) + target_counts[bid] + num_non_answers)
        ans1_num_non_answers_used = int(math.ceil(ans1_percent_non_answer_used * num_non_answers))
        ans1_na_indexes = na_indexes[0:ans1_num_non_answers_used]
        answer1_indexes += ans1_na_indexes
        ans2_na_indexes = na_indexes[ans1_num_non_answers_used:]
        answer2_indexes += ans2_na_indexes

        # 3c. choosing noise1 indexes
        noise1_num_answers_used = int(round(noise_percent_answer_used * len(target_fld_bin_vals[bid][0:target_counts[bid]])))
        noise1_num_non_answers_used = int(round(noise_percent_non_answer_used * len(target_fld_bin_vals[bid][target_counts[bid]:])))
        noise1_indexes += range(len(target_fld_vals), len(target_fld_vals) + noise1_num_answers_used)
        noise1_indexes += range(len(target_fld_vals) + len(target_fld_bin_vals[bid]),
                                len(target_fld_vals) + len(target_fld_bin_vals[bid]) + noise1_num_non_answers_used)

        # 3d. add current bin's target vals to final target vals signal
        target_fld_vals += target_fld_bin_vals[bid]

    # 4. fill in synthetic answer's values
    cur_ans1_val = 0
    cur_ans2_val = 0
    answer1_vals = []
    answer2_vals = []
    answer1_indexes = set(answer1_indexes)
    answer2_indexes = set(answer2_indexes)
    for i in xrange(len(df)):
        if i in answer1_indexes:
            ans1_val = cur_ans1_val
            cur_ans1_val += 1
        else:
            ans1_val = 99999999
        answer1_vals.append(ans1_val)

        if i in answer2_indexes:
            ans2_val = cur_ans2_val
            cur_ans2_val += 1
        else:
            ans2_val = 99999999
        answer2_vals.append(ans2_val)

    # 5. create noise column
    cur_noise1_val = 0
    noise1_vals = []
    noise1_indexes = set(noise1_indexes)
    for i in xrange(len(df)):
        if i in noise1_indexes:
            noise1_val = cur_noise1_val
            cur_noise1_val += 1
        else:
            noise1_val = 99999999
        noise1_vals.append(noise1_val)

    # 6. adding synthetic answer's preds
    ans_preds = [
        ('answer1', '<', cur_ans1_val),
        ('answer2', '<', cur_ans2_val),
    ]
    noise_preds = [
        ('noise1', '<', cur_noise1_val),
    ]
    return target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, ans_preds, noise_preds


def thres_rand_dep_filter(df, meta, dep_scores, bin_pred_probs, bin_pred_pair_probs,
                          pred_pair_counts, pred_dep_percent,
                          sorting_random_seed, slack_percent, system, debug=False, preds=None):
    ## 1. determine valid pairs and in-bound pairs ##
    start = timer()
    target_bounds = get_target_count_bounds(meta['target_counts'], slack_percent)
    valid_pairs = {}
    in_bounds_pairs = {}
    at_b3p = 0
    for (pid1, pid2), b3p in bin_pred_pair_probs.iteritems():
        # checking valid counts
        pp_bin_counts = [b3p[bid] * float(bc) for bid, bc in enumerate(meta['bin_counts'])]
        valid = True
        for bid, (lower, upper) in enumerate(target_bounds):
            if pp_bin_counts[bid] < lower or pp_bin_counts[bid] > upper:
                valid = False
                break
        if valid:
            if debug:
                print "- valid at_b3p={}, valid={} + {}".format(at_b3p, preds[pid1], preds[pid2])
            valid_pairs[(pid1, pid2)] = None

        # checking indep bounds
        pp_bin_bounds = get_target_count_bounds(pp_bin_counts, slack_percent)
        bpp1 = bin_pred_probs[pid1]
        bpp2 = bin_pred_probs[pid2]
        within_indep_bounds = True
        for bid, (lower, upper) in enumerate(pp_bin_bounds):
            ppc = int(round(bpp1[bid] * bpp2[bid] * meta['bin_counts'][bid]))
            if ppc < lower or ppc > upper:
                within_indep_bounds = False
                break
        if within_indep_bounds:
            in_bounds_pairs[(pid1, pid2)] = None

        at_b3p += 1
        if at_b3p % 3000000 == 0:
            print "{} - at {} of {} (valid_pairs={})"\
                .format(dt.datetime.now(), at_b3p, len(bin_pred_pair_probs), len(valid_pairs))
    print "done w/ determining valid / in-bound pairs ({} sec)".format(timer() - start)
    print "len bin_pred_pair_probs = {}, len valid_pairs = {}, len input dep_scores = {}".format(len(bin_pred_pair_probs), len(valid_pairs), len(dep_scores))

    ## 2. determining percent to go into dep_scores ##
    meta['num_dep_scores_used'] = int(round(len(pred_pair_counts) * pred_dep_percent))
    if system in ("emerilThresRandIndep", "emerilTriHybrid"):
        meta['dep_scores_num_indep'] = int(math.ceil(0.5 * meta['num_dep_scores_used']))
        meta['dep_scores_num_rand'] = int(math.floor(0.5 * meta['num_dep_scores_used']))
    elif system in ("emerilThresIndep", "emerilTiHybrid"):
        meta['dep_scores_num_indep'] = meta['num_dep_scores_used']
        meta['dep_scores_num_rand'] = 0
    elif system == "emerilThresRand":
        meta['dep_scores_num_indep'] = 0
        meta['dep_scores_num_rand'] = meta['num_dep_scores_used']

    ## 3. getting dep scores for in-bounds and out-of-bounds dep scores ##
    in_bounds_scores = []
    out_bounds_scores = []
    out_bounds_dep_scores = []
    for cur_rank, (score, pid1, pid2) in enumerate(dep_scores):
        is_in_bounds = (pid1, pid2) in in_bounds_pairs
        if debug:
            if (pid1, pid2) in valid_pairs:
                print " - cur_rank={}, valid score={:.3f}, in_bound={}, out_bounds_rank={}, preds={} + {}".format(cur_rank, score, is_in_bounds, len(out_bounds_scores), preds[pid1], preds[pid2])
        # noting scores not in bounds
        if is_in_bounds:
            in_bounds_scores.append(score)
        else:
            out_bounds_scores.append(score)
            out_bounds_dep_scores.append((score, pid1, pid2))
    print "in bounds (total={}): min/mean/median/max: {:.4f}/{:.4f}/{:.4f}/{:.4f}"\
        .format(len(in_bounds_scores),
                min(in_bounds_scores) if in_bounds_scores else -1,
                np.mean(in_bounds_scores) if in_bounds_scores else -1,
                np.median(in_bounds_scores) if in_bounds_scores else -1,
                max(in_bounds_scores) if in_bounds_scores else -1)
    print "out bounds (total={}): min/mean/median/max: {:.4f}/{:.4f}/{:.4f}/{:.4f}"\
        .format(len(out_bounds_scores),
                min(out_bounds_scores) if out_bounds_scores else -1,
                np.mean(out_bounds_scores) if out_bounds_scores else -1,
                np.median(out_bounds_scores) if out_bounds_scores else -1,
                max(out_bounds_scores) if out_bounds_scores else -1)

    ## 4. for system emerilThresRandIndep/etc, adding indep scores to start ##
    if meta['dep_scores_num_indep'] > 0:
        final_dep_scores = out_bounds_dep_scores[0:meta['dep_scores_num_indep']]
        del out_bounds_dep_scores[0:meta['dep_scores_num_indep']]
    else:
        final_dep_scores = []

    ## 5. adding random out-bounds dep scores to final list ##
    np_random = np.random.RandomState(sorting_random_seed)
    np_random.shuffle(out_bounds_dep_scores)
    final_dep_scores += out_bounds_dep_scores[0:meta['dep_scores_num_rand']]

    ## 6. counting number of valid in final list and returning ##
    meta['dep_scores_num_valid'] = 0
    for cur_rank, (score, pid1, pid2) in enumerate(final_dep_scores):
        if (pid1, pid2) in valid_pairs:
            meta['dep_scores_num_valid'] += 1
    print "returning {} dep scores ({} valid, {} indep)".format(len(final_dep_scores), meta['dep_scores_num_valid'], meta['dep_scores_num_indep'])
    return final_dep_scores



###############################################################################
################################## Test Runs ##################################

def update_v1_meta_with_v3_data(syn_data_mode, df, meta, preds, random_seed,
                                pred_pair_counts, bin_pred_probs, bin_pred_pair_probs):
    target_fld_bin_vals, bin_counts, adjusted_bin_edges = \
            get_target_fld_vals_from_df(df, meta['target_fld'], random_seed)

    np_random = np.random.RandomState(random_seed)

    # Note: v1b: (greedy 50/50, emIndep good)
    #       v1c: (greedy worse, emIndep good)
    print("syn_data_mode = {}".format(syn_data_mode))
    if syn_data_mode  == "v1b":
        percent_non_answer_used = 0.15
        ans1_percent_non_answer_used = 0.5
        noise_percent_answer_used = 0.3
        noise_percent_non_answer_used = bounded_normal_draw(np_random, loc=0.2, scale=0.1, min_val=0.1, max_val=0.5)
    elif syn_data_mode == "v1c":
        percent_non_answer_used = 0.7
        ans1_percent_non_answer_used = 0.5
        noise_percent_answer_used = 0.9
        noise_percent_non_answer_used = 0.1
    target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, answer_preds, noise_preds = \
            get_synthetic_answer_v3(df, target_fld_bin_vals, bin_counts, print_details=False,
                                    percent_non_answer_used=percent_non_answer_used,
                                    ans1_percent_non_answer_used=ans1_percent_non_answer_used,
                                    noise_percent_answer_used=noise_percent_answer_used,
                                    noise_percent_non_answer_used=noise_percent_non_answer_used)

    # . adding additional columns, changing answer columns
    df['noise1'] = noise1_vals
    preds += noise_preds

    print "Pred debugging (target = {}):".format(meta['target_counts'])
    for pid in list(meta['answer_pids']) + list(range(len(preds) - 1, len(preds))):
        d = np.histogram(df.query(Query.get_qs_pred(*preds[pid]))[meta['target_fld']], meta['adjusted_bin_edges'])[0]
        signal_dist = np.sum(np.abs(meta['target_counts'] - d))
        print("- pid={}, corr={:.3f}, signal_dist={:.0f}, pred: {}, d: {}"
              .format(pid, pearsonr(meta['target_counts'], d)[0], signal_dist, preds[pid], d))
    print "Answer preds: {}".format(meta['answer_pids'])

    # updating pred pair meta
    if pred_pair_counts:
        new_pred_ids = [len(preds) - 1]
        update_pred_pair_meta_with_new_preds(new_pred_ids, df, preds, meta, bin_counts,
                                             target_fld_vals, pred_pair_counts, bin_pred_probs,
                                             bin_pred_pair_probs)


def update_v2_meta_with_v3_data(syn_data_mode, df, meta, preds, random_seed,
                                pred_pair_counts, bin_pred_probs, bin_pred_pair_probs):
    target_fld_bin_vals, bin_counts, adjusted_bin_edges = \
            get_target_fld_vals_from_df(df, meta['target_fld'], random_seed)

    np_random = np.random.RandomState(random_seed)
    # Note: : (greedy 50/50, emIndep good)
    #       v2c: (greedy good, emIndep okay)
    if syn_data_mode == "v2b":
        percent_non_answer_used = 0.3
        ans1_percent_non_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.2, min_val=0.1, max_val=0.9)
        noise_percent_answer_used = 0.8
        noise_percent_non_answer_used = 0.1
    elif syn_data_mode == "v2c":
        percent_non_answer_used = 0.1
        ans1_percent_non_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.3, min_val=0.1, max_val=0.9)
        print("NOTE: ans1_percent_non_answer_used={}".format(ans1_percent_non_answer_used))
        noise_percent_answer_used = 0.7
        noise_percent_non_answer_used = 0.1
    elif syn_data_mode == "v9b":
        percent_non_answer_used = 0.3
        ans1_percent_non_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.2, min_val=0.1, max_val=0.9)
        noise_percent_answer_used = 0.8
        noise_percent_non_answer_used = 0.1
    elif syn_data_mode == "v9c":
        percent_non_answer_used = 0.1
        ans1_percent_non_answer_used = bounded_normal_draw(np_random, loc=0.5, scale=0.3, min_val=0.1, max_val=0.9)
        print("NOTE: ans1_percent_non_answer_used={}".format(ans1_percent_non_answer_used))
        noise_percent_answer_used = 0.7
        noise_percent_non_answer_used = 0.1
    target_counts, target_fld_vals, answer1_vals, answer2_vals, noise1_vals, answer_preds, noise_preds = \
            get_synthetic_answer_v3(df, target_fld_bin_vals, bin_counts, print_details=False,
                                    percent_non_answer_used=percent_non_answer_used,
                                    ans1_percent_non_answer_used=ans1_percent_non_answer_used,
                                    noise_percent_answer_used=noise_percent_answer_used,
                                    noise_percent_non_answer_used=noise_percent_non_answer_used)

    # . adding additional columns, changing answer columns
    df['answer3'] = answer1_vals
    df['answer4'] = answer2_vals
    df['noise2'] = noise1_vals
    answer_preds[0] = ('answer3', answer_preds[0][1], answer_preds[0][2])
    answer_preds[1] = ('answer4', answer_preds[1][1], answer_preds[1][2])
    noise_preds[0] = ('noise2', noise_preds[0][1], noise_preds[0][2])
    #del preds[0:3]  # removing old synthethic-answer preds
    preds += answer_preds + noise_preds
    meta['answer_preds'] = answer_preds
    meta['answer_pids'] = [len(preds) - 3, len(preds) - 2]
    meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)
    qs = Query.get_pandas_query_from_preds(answer_preds)
    meta['answer_counts'] = target_counts #np.histogram(df.query(qs)[meta['target_fld']], meta['adjusted_bin_edges'])[0]
    meta['target_counts'] = target_counts #meta['answer_counts']

    print "Pred debugging:"
    for pid in [0, 178] + range(len(preds) - 6, len(preds)):
        d = np.histogram(df.query(Query.get_qs_pred(*preds[pid]))[meta['target_fld']], meta['adjusted_bin_edges'])[0]
        signal_dist = np.sum(np.abs(meta['target_counts'] - d))
        print("- pid={}, corr={:.3f}, signal_dist={:.0f}, pred: {}"
              .format(pid, pearsonr(meta['target_counts'], d)[0], signal_dist, preds[pid]))
    print "Answer preds: {}".format(meta['answer_pids'])

    # updating pred pair meta
    if pred_pair_counts:
        new_pred_ids = [len(preds) - 3, len(preds) - 2, len(preds) - 1]
        update_pred_pair_meta_with_new_preds(new_pred_ids, df, preds, meta, bin_counts,
                                             target_fld_vals, pred_pair_counts, bin_pred_probs,
                                             bin_pred_pair_probs)





def update_meta_with_user_sims(syn_data_mode, df, meta, preds,
                               pred_pair_counts, bin_pred_probs, bin_pred_pair_probs):
    # 1. creating valid answer with wrong field names
    a1, a2 = meta['answer_preds']
    a1_vals = df[a1[0]].tolist()
    a2_vals = df[a2[0]].tolist()
    df['bad1_{}'.format(a1[0])] = a1_vals
    df['bad1_{}'.format(a2[0])] = a2_vals
    new_ans_preds = []
    new_ans_preds.append(('bad1_{}'.format(a1[0]), a1[1], a1[2]))
    new_ans_preds.append(('bad1_{}'.format(a2[0]), a2[1], a2[2]))

    # 2. adding user preds with same column as answer_preds, but wrong vals
    user_preds = []
    user_pred1 = (a1[0], a1[1], 500)
    user_pred2 = (a2[0], a2[1], 500)
    user_preds.append(user_pred1)
    user_preds.append(user_pred2)

    # 3. adding new preds to preds
    preds += new_ans_preds
    new_ans_pred_ids = [len(preds) - i for i in xrange(len(new_ans_preds), 0, -1)]
    preds += user_preds
    user_pred_ids = [len(preds) - i for i in xrange(len(user_preds), 0, -1)]
    print "user_pred_ids = {}".format(user_pred_ids)
    new_pred_ids = new_ans_pred_ids + user_pred_ids
    print "Relevant preds after code-sim change:"
    for pid in list(meta['answer_pids']) + list(new_pred_ids):
        print " - pid={}, pred: {}".format(pid, preds[pid])

    # 4. add user preds to meta
    meta['user_pred_ids'] = user_pred_ids
    meta['user_preds'] = user_preds

    # 5. updating pred pair meta
    if pred_pair_counts:
        update_pred_pair_meta_with_new_preds(new_pred_ids, df, preds, meta,
                                             meta['bin_counts'],
                                             df[meta['target_fld']].tolist(),
                                             pred_pair_counts,
                                             bin_pred_probs,
                                             bin_pred_pair_probs)


def get_syn_data_and_pred_pair_meta(num_rows, num_columns, num_preds, corr,
                                    random_seed, system, syn_data_mode,
                                    print_details=False, min_print_details=True,
                                    n_processes=1, skip_loading=False,
                                    use_code_sim=False):
    # 1. load/generate synthetic data with synthetic answer
    start = timer()
    data_cache_key = get_data_cache_key(syn_data_mode, num_rows, num_columns, num_preds, corr, random_seed)
    if syn_data_mode in ("v1", "v1b", "v1c"):
        data_cache_file = os.path.join(DATA_GEN_CACHE_DIR, "v1", data_cache_key + ".cache")
        df, preds, meta = get_synthetic_data_and_answer_v1(num_rows, num_columns, num_preds,
                                                           corr, random_seed,
                                                           data_cache_file,
                                                           print_details=print_details)
        ppm_syn_data_mode = "v1"
    elif syn_data_mode in ("v2", "v2b", "v2c"):
        data_cache_file = os.path.join(DATA_GEN_CACHE_DIR, "v2", data_cache_key + ".cache")
        df, preds, meta = get_synthetic_data_and_answer_v2(num_rows, num_columns,
                                                           num_preds, random_seed,
                                                           data_cache_file,
                                                           print_details=print_details)
        ppm_syn_data_mode = "v2"
    elif syn_data_mode in ("v9", "v9b", "v9c", "v8", "v8b", "v8c", "v7", "v7b", "v7c"):
        data_cache_file = os.path.join(DATA_GEN_CACHE_DIR, "v9", data_cache_key + ".cache")
        if print_details:
            print "Loading data cache file: {}".format(data_cache_file)
        df, preds, meta = get_synthetic_data_and_answer_v9(num_rows, num_columns, num_preds,
                                                           corr, random_seed,
                                                           data_cache_file,
                                                           print_details=print_details)
        ppm_syn_data_mode = "v9"
    if min_print_details or print_details:
        print("RUNTIME: load/generate synthetic data: {:.3f} sec".format(timer() - start))

    # 2. load/generate dep_score meta, pred/pred-pair probs
    if system in ('tiresias', 'emerilRandom', 'emerilIndep', 'emerilHybrid', 'emerilThresRand', 'emerilThresRandIndep', 'emerilTriHybrid', "emerilTiHybrid", 'emerilThresIndep'):
        start = timer()
        pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs = \
            get_pred_pair_meta(df, preds, meta['target_fld'], meta['adjusted_bin_edges'],
                               ppm_syn_data_mode, data_cache_key, print_details=print_details,
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

    # 3. switching synthetic answer if rw_data_mode == "11"
    if syn_data_mode in ("v1", "v2", "v9"):
        pass  # we leave answers as is
    elif syn_data_mode in ("v1b", "v1c"):
        start = timer()
        update_v1_meta_with_v3_data(syn_data_mode, df, meta, preds, random_seed,
                                    pred_pair_counts, bin_pred_probs, bin_pred_pair_probs)
        if min_print_details or print_details:
            print("RUNTIME: {} data update ({} preds): {:.3f} sec"
                  .format(syn_data_mode, len(preds), timer() - start))
    elif syn_data_mode in ("v2b", "v2c", "v9b", "v9c"):
        start = timer()
        update_v2_meta_with_v3_data(syn_data_mode, df, meta, preds, random_seed,
                                    pred_pair_counts, bin_pred_probs, bin_pred_pair_probs)
        if min_print_details or print_details:
            print("RUNTIME: {} data update ({} preds): {:.3f} sec"
                  .format(syn_data_mode, len(preds), timer() - start))
    elif syn_data_mode in ("v8", "v8b", "v8c"):
        start = timer()
        if print_details:
            print "Starting v9->v8 conversion"
        update_v9_meta_with_v8_data(corr, syn_data_mode, df, meta, preds, random_seed,
                                    pred_pair_counts, bin_pred_probs, bin_pred_pair_probs)
        if min_print_details or print_details:
            print("RUNTIME: {} v9-v8 update ({} preds): {:.3f} sec"
                  .format(syn_data_mode, len(preds), timer() - start))
    elif syn_data_mode in ("v7", "v7b", "v7c"):
        start = timer()
        if print_details:
            print "Starting v9->v7 conversion"
        update_v9_meta_with_v7_data(corr, syn_data_mode, df, meta, preds, random_seed,
                                    pred_pair_counts, bin_pred_probs, bin_pred_pair_probs)
        if min_print_details or print_details:
            print("RUNTIME: {} v9->v7 update ({} preds): {:.3f} sec"
                  .format(syn_data_mode, len(preds), timer() - start))
    else:
        raise Exception("bad data mode")

    # 4. if code sim, adding more answers and preds
    if use_code_sim:
        update_meta_with_user_sims(syn_data_mode, df, meta, preds,
                                   pred_pair_counts, bin_pred_probs, bin_pred_pair_probs)

    return df, preds, meta, pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs










###############################################################################
# syndata v8
###############################################################################

def test_syn_answer_corr(bid, answer1_indexes, answer2_indexes, target_fld_vals, target_fld, adjusted_bin_edges):
    cur_ans1_val = 0
    cur_ans2_val = 0
    answer1_vals = []
    answer2_vals = []
    answer1_indexes_set = set(answer1_indexes)
    answer2_indexes_set = set(answer2_indexes)
    for i in xrange(len(target_fld_vals)):
        if i in answer1_indexes_set:
            ans1_val = cur_ans1_val
            cur_ans1_val += 1
        else:
            ans1_val = 99999999
        answer1_vals.append(ans1_val)

        if i in answer2_indexes_set:
            ans2_val = cur_ans2_val
            cur_ans2_val += 1
        else:
            ans2_val = 99999999
        answer2_vals.append(ans2_val)
    ans_preds = [
        ('answer1', '<', cur_ans1_val),
        ('answer2', '<', cur_ans2_val),
    ]

    data = []
    for i in xrange(len(answer1_vals)):
        data.append([target_fld_vals[i], answer1_vals[i], answer2_vals[i]])
    df = pd.DataFrame.from_records(data, columns=[target_fld, 'answer1', 'answer2'])
    p1, p2 = ans_preds
    df_c1 = df.query(Query.get_pandas_query_from_preds([p1]))
    df_c2 = df.query(Query.get_pandas_query_from_preds([p2]))
    df_cc = df.query(Query.get_pandas_query_from_preds(ans_preds))
    hist1 = np.histogram(df_c1[target_fld], adjusted_bin_edges)[0]
    hist2 = np.histogram(df_c2[target_fld], adjusted_bin_edges)[0]
    hist_cc = np.histogram(df_cc[target_fld], adjusted_bin_edges)[0]
#     print "- hist1: {}".format(list(hist1))
#     print "- hist2: {}".format(list(hist2))
#     print "- hist_cc: {}".format(hist_cc)
    return pearsonr(hist1[0:bid], hist2[0:bid])[0]



def get_signal_dist(s1, s2):
    return sum([abs(s1[i] - s2[i]) for i in xrange(len(s1))])


def eval_syn_answer(target_fld_vals, target_fld, adjusted_bin_edges, target_counts,
                    answer1_indexes, answer2_indexes, honeypot_indexes):
    # 4. fill in synthetic answer's values
    cur_ans1_val = 0
    cur_ans2_val = 0
    answer1_vals = []
    answer2_vals = []
    answer1_indexes = set(answer1_indexes)
    answer2_indexes = set(answer2_indexes)
    for i in xrange(len(target_fld_vals)):
        if i in answer1_indexes:
            ans1_val = cur_ans1_val
            cur_ans1_val += 1
        else:
            ans1_val = 99999999
        answer1_vals.append(ans1_val)

        if i in answer2_indexes:
            ans2_val = cur_ans2_val
            cur_ans2_val += 1
        else:
            ans2_val = 99999999
        answer2_vals.append(ans2_val)

    # 5. create noise column
    cur_honeypot_val = 0
    honeypot_vals = []
    honeypot_indexes = set(honeypot_indexes)
    for i in xrange(len(target_fld_vals)):
        if i in honeypot_indexes:
            honeypot_val = cur_honeypot_val
            cur_honeypot_val += 1
        else:
            honeypot_val = 99999999
        honeypot_vals.append(honeypot_val)

    # 6. adding synthetic answer's preds
    answer_preds = [
        ('answer1', '<', 99999999),
        ('answer2', '<', 99999999),
    ]
    honeypot_preds = [
        ('honeypot', '<', 99999999),
    ]

    eval_syn_answer_vals(target_fld_vals, answer1_vals, answer2_vals, honeypot_vals,
                         target_fld, adjusted_bin_edges, answer_preds, honeypot_preds)


def eval_syn_answer_vals(target_fld_vals, target_fld, adjusted_bin_edges, target_counts,
                         answer1_vals, answer2_vals, honeypot_vals,
                         answer_preds, honeypot_preds):
    data = []
    for i in xrange(len(answer1_vals)):
        data.append([target_fld_vals[i], answer1_vals[i], answer2_vals[i], honeypot_vals[i]])
        cols = [target_fld, answer_preds[0][0], answer_preds[1][0], honeypot_preds[0][0]]
    df = pd.DataFrame.from_records(data, columns=cols)
    p1, p2 = answer_preds
    df_c1 = df.query(Query.get_pandas_query_from_preds([p1]))
    df_c2 = df.query(Query.get_pandas_query_from_preds([p2]))
    df_honey = df.query(Query.get_pandas_query_from_preds(honeypot_preds))
    df_cc = df.query(Query.get_pandas_query_from_preds(answer_preds))

    hist1 = np.histogram(df_c1[target_fld], adjusted_bin_edges)[0]
    hist2 = np.histogram(df_c2[target_fld], adjusted_bin_edges)[0]
    hist_cc = np.histogram(df_cc[target_fld], adjusted_bin_edges)[0]
    hist_honey = np.histogram(df_honey[target_fld], adjusted_bin_edges)[0]

    corr1 = pearsonr(hist1, target_counts)[0]
    signal_dist1 = get_signal_dist(hist1, target_counts)
    corr2 = pearsonr(hist2, target_counts)[0]
    signal_dist2 = get_signal_dist(hist2, target_counts)
    corr_honey = pearsonr(hist_honey, target_counts)[0]
    signal_dist_honey = get_signal_dist(hist_honey, target_counts)

    print "- hist1: {}, corr={:.3f}, signal_dist={:.3f}".format(list(hist1), corr1, signal_dist1)
    print "- hist2: {}, corr={:.3f}, signal_dist={:.3f}".format(list(hist2), corr2, signal_dist2)
    print "- honey: {}, corr={:.3f}, signal_dist={:.3f}".format(list(hist_honey), corr_honey, signal_dist_honey)
    print "- hist_cc: {}".format(hist_cc)
    print "- target_counts: {}".format(target_counts)
    col_corr = pearsonr(df[p1[0]], df[p2[0]])[0]
    sdists = [(signal_dist_honey, "honey"), (signal_dist1, "ans1", (signal_dist2, "ans2"))]
    sdists.sort()
    best_sdist = sdists[0][1]
    print "- corr(hist1, hist2) = {:.3f}, col_corr={:.3f}, best sdist={}".format(pearsonr(hist1, hist2)[0], col_corr, best_sdist)



def adjust_syn_answer_corrs(orig_agg_indexes, target_corr, bin_counts,
                            target_fld_vals, target_fld, adjusted_bin_edges,
                            print_details=False, max_adjustments=50):
    agg_indexes = copy.deepcopy(orig_agg_indexes)

    # re-building answers since it changed above
    answer1_indexes = []
    answer2_indexes = []
    for bid in xrange(len(bin_counts)):
        answer1_indexes += agg_indexes['answer'][bid] + agg_indexes['noise1'][bid]
        answer2_indexes += agg_indexes['answer'][bid] + agg_indexes['noise2'][bid]

    # attempting to get corr close to goal
    for i in xrange(max_adjustments):
        if print_details:
            print "\n===== run {} ===========".format(i)
            for key in agg_indexes:
                cur_counts = [len(cur_indexes) for cur_indexes in agg_indexes[key]]
                print "{}: {}".format(key, cur_counts)

        # 1. getting current correlation
        cur_corr = test_syn_answer_corr(len(bin_counts), answer1_indexes, answer2_indexes,
                                        target_fld_vals, target_fld, adjusted_bin_edges)
        if print_details:
            print("cur_corr={:.3f}".format(cur_corr))

        # 4. otherwise, terminate (since within target correlation range)
        if abs(cur_corr - target_corr) < 0.1 or (target_corr * 0.8 <= cur_corr <= target_corr * 1.2):
            if print_details:
                print("Corr close to target_corr: cur={:.3f}".format(cur_corr))
            break

        # 2. if corr too low, shrink biggest gap between noise1 & noise2
        elif cur_corr < target_corr * 0.8:
            # find largest gap
            largest_diff = None
            for bid in xrange(len(bin_counts)):
                cur_diff = abs(len(agg_indexes['noise1'][bid]) - len(agg_indexes['noise2'][bid]))
                larger = 'noise1' if len(agg_indexes['noise1'][bid]) > len(agg_indexes['noise2'][bid]) else 'noise2'
                smaller = 'noise2' if len(agg_indexes['noise1'][bid]) > len(agg_indexes['noise2'][bid]) else 'noise1'
                dec_amount = int(round(0.2 * len(agg_indexes[larger][bid])))
                if (largest_diff is None or cur_diff > largest_diff['cur_diff']) and dec_amount > 0:
                    largest_diff = {
                        'cur_diff': cur_diff,
                        'targ_bid': bid,
                        'larger_index': larger,
                        'smaller_index': smaller,
                        'dec_amount': dec_amount,
                    }
            if largest_diff is None:
                if print_details:
                    print("No valid largest_diff; ending")
                break
            else:
                bid = largest_diff['targ_bid']
                dec_amount = largest_diff['dec_amount']
                li = largest_diff['larger_index']
                si = largest_diff['smaller_index']
                if print_details:
                    print("Shrinking largest: decreasing {} bid={} by {}".format(li, bid, dec_amount))
                agg_indexes[si][bid] += agg_indexes[li][bid][0:dec_amount]
                del agg_indexes[li][bid][0:dec_amount]

        # 3. if corr too high, expand smallest gap between noise1 & noise2
        elif cur_corr > target_corr * 1.2:
            # find smallest gap
            smallest_diff = None
            for bid in xrange(len(bin_counts)):
                cur_diff = abs(len(agg_indexes['noise1'][bid]) - len(agg_indexes['noise2'][bid]))
                larger = 'noise1' if len(agg_indexes['noise1'][bid]) > len(agg_indexes['noise2'][bid]) else 'noise2'
                smaller = 'noise2' if len(agg_indexes['noise1'][bid]) > len(agg_indexes['noise2'][bid]) else 'noise1'
                dec_amount = int(round(0.8 * len(agg_indexes[smaller][bid])))
                if (smallest_diff is None or cur_diff > smallest_diff['cur_diff']) and dec_amount > 0:
                    smallest_diff = {
                        'cur_diff': cur_diff,
                        'targ_bid': bid,
                        'larger_index': larger,
                        'smaller_index': smaller,
                        'dec_amount': dec_amount,
                    }
            if smallest_diff is None:
                if print_details:
                    print("No valid smallest_diff; ending")
                break
            else:
                bid = smallest_diff['targ_bid']
                dec_amount = smallest_diff['dec_amount']
                li = smallest_diff['larger_index']
                si = smallest_diff['smaller_index']
                if print_details:
                    print("Expanding smallest: decreasing {} bid={} by {}".format(si, bid, dec_amount))
                agg_indexes[li][bid] += agg_indexes[si][bid][0:dec_amount]
                del agg_indexes[si][bid][0:dec_amount]

        # 5. re-building answers since it changed above
        answer1_indexes = []
        answer2_indexes = []
        for bid in xrange(len(bin_counts)):
            answer1_indexes += agg_indexes['answer'][bid] + agg_indexes['noise1'][bid]
            answer2_indexes += agg_indexes['answer'][bid] + agg_indexes['noise2'][bid]

    return answer1_indexes, answer2_indexes



def get_v8_syn_answer(num_rows, corr, target_fld = 'target_fld', print_details=False,

                      # this is the percent of rows outside of answer
                      # tuples to throw off individ corr
                      percent_noise_used=0.7,

                      # what percent of the above percent goes to
                      # answer1; remainder to answer2
                      percent_noise_used_ans1=0.5,

                      # percent of answer and non-answer used by
                      # noise column
                      honeypot_percent_answer_used=0.7,
                      honeypot_percent_noise_used=0.1):

    # 1c. getting synthetic answer and meta
    target_fld_bin_vals, bin_counts, adjusted_bin_edges = get_target_fld_vals(num_rows)

    # 1c. building target column values and target signal
    target_counts = np.zeros(len(bin_counts), dtype=int)
    for index in xrange(len(target_counts)):
        if index in (0, 1, len(target_counts)-2, len(target_counts)-1):
            target_counts[index] = int(round(0.9 * bin_counts[index]))
        else:
            target_counts[index] = int(round(0.0 * bin_counts[index]))

    ## 2. adjusting syn answer corrs ##
    if print_details:
        print "target corr: {}".format(corr)
    target_fld_vals = []
    answer1_indexes = []
    answer2_indexes = []
    honeypot_indexes = []
    invert_pnau = False
    agg_indexes = defaultdict(list)
    for bid in xrange(len(bin_counts)):
        # 2a. adjusting ans1_percent_non_answer_used based on current corr
        if bid > 0:
            cur_corr = test_syn_answer_corr(bid, answer1_indexes, answer2_indexes,
                                            target_fld_vals, target_fld, adjusted_bin_edges)
            if cur_corr > corr * 1.2:
                percent_noise_used_ans1 = 0.87
                invert_pnau = not invert_pnau
            else:
                percent_noise_used_ans1 = 0.5
        else:
            cur_corr = -1

        # 2b. separating available indexes
        bin_indexes = range(len(target_fld_vals), len(target_fld_vals) + len(target_fld_bin_vals[bid]))
        bin_answer_indexes = bin_indexes[0:target_counts[bid]]
        num_noise_used = int(round(percent_noise_used * len(bin_indexes)))
        bin_noise_indexes = bin_indexes[target_counts[bid] : target_counts[bid] + num_noise_used]

        # 2c. adding bin's target fld values
        target_fld_vals += target_fld_bin_vals[bid]

        # 2d. noting answers
        answer1_indexes += bin_answer_indexes
        answer2_indexes += bin_answer_indexes

        # 2e. determining num noise to use for ans1 & ans2
        cur_percent_noise_ans1 = percent_noise_used_ans1 if invert_pnau else 1 - percent_noise_used_ans1
        num_noise_used_ans1 = int(math.ceil(cur_percent_noise_ans1 * num_noise_used))
        bin_noise1_indexes = bin_noise_indexes[0:num_noise_used_ans1]
        bin_noise2_indexes = bin_noise_indexes[num_noise_used_ans1:]
        answer1_indexes += bin_noise1_indexes
        answer2_indexes += bin_noise2_indexes

        agg_indexes['answer'].append(bin_answer_indexes)
        agg_indexes['noise1'].append(bin_noise1_indexes)
        agg_indexes['noise2'].append(bin_noise2_indexes)


        # 2f. choosing honeypot indexes
        num_honeypot_ans_used = int(round(honeypot_percent_answer_used * len(bin_answer_indexes)))
        num_honeypot_noise_used = int(round(honeypot_percent_noise_used * len(bin_noise_indexes)))
        honeypot_indexes += bin_answer_indexes[0:num_honeypot_ans_used] + bin_noise_indexes[0:num_honeypot_noise_used]

        # 2g. debug info
        if print_details:
            print("{}. cur_corr={:.3f}, avail={}, ans={}, noise={}, noise1={}, noise2={}"
                  .format(bid, cur_corr, len(bin_indexes), len(bin_answer_indexes),
                          len(bin_noise_indexes), len(bin_noise1_indexes), len(bin_noise2_indexes)))

    # print "\n## BEFORE FIX: ##"
    # eval_syn_answer(target_fld_vals, target_fld, adjusted_bin_edges, target_counts,
    #                 answer1_indexes, answer2_indexes, honeypot_indexes)

    ## 3. adjusting corr ##
    answer1_indexes, answer2_indexes = \
        adjust_syn_answer_corrs(agg_indexes, corr, bin_counts,
                                target_fld_vals, target_fld, adjusted_bin_edges,
                                print_details=print_details)

    ## 4. returning data ##
    return (target_fld, target_fld_vals, bin_counts, target_counts,
            adjusted_bin_edges, answer1_indexes, answer2_indexes, honeypot_indexes)


def update_v9_meta_with_v8_data(corr, syn_data_mode, df, meta, preds, random_seed,
                                pred_pair_counts, bin_pred_probs, bin_pred_pair_probs):

    # getting data and meta data
    np_random = np.random.RandomState(random_seed)
    if syn_data_mode == "v8":
        percent_noise_used = 0.4
        percent_noise_used_ans1 = bounded_normal_draw(np_random, loc=0.5, scale=0.3, min_val=0.1, max_val=0.9)
        honeypot_percent_answer_used = bounded_normal_draw(np_random, loc=0.3, scale=0.1, min_val=0.1, max_val=0.9)
        honeypot_percent_noise_used = 0.6
    elif syn_data_mode == "v8b":
        percent_noise_used = bounded_normal_draw(np_random, loc=0.4, scale=0.1, min_val=0.1, max_val=0.9)
        percent_noise_used_ans1 = 0.5
        honeypot_percent_answer_used = bounded_normal_draw(np_random, loc=0.3, scale=0.1, min_val=0.1, max_val=0.9)
        honeypot_percent_noise_used = bounded_normal_draw(np_random, loc=0.3, scale=0.1, min_val=0.1, max_val=0.9)
    elif syn_data_mode == "v8c":
        percent_noise_used = 0.3
        percent_noise_used_ans1 = bounded_normal_draw(np_random, loc=0.5, scale=0.2, min_val=0.1, max_val=0.9)
        honeypot_percent_answer_used = 0.7
        honeypot_percent_noise_used = 0.3

    (target_fld, target_fld_vals, bin_counts, target_counts, adjusted_bin_edges,
     answer1_indexes, answer2_indexes, honeypot_indexes) = \
        get_v8_syn_answer(len(df), corr,
                          target_fld='target_fld',
                          percent_noise_used=percent_noise_used,
                          percent_noise_used_ans1=percent_noise_used_ans1,
                          honeypot_percent_answer_used=honeypot_percent_answer_used,
                          honeypot_percent_noise_used=honeypot_percent_noise_used)

    # creating answer columns
    cur_ans1_val = 0
    cur_ans2_val = 0
    answer1_vals = []
    answer2_vals = []
    answer1_indexes = set(answer1_indexes)
    answer2_indexes = set(answer2_indexes)
    for i in xrange(len(target_fld_vals)):
        if i in answer1_indexes:
            ans1_val = cur_ans1_val
            cur_ans1_val += 1
        else:
            ans1_val = 99999999
        answer1_vals.append(ans1_val)

        if i in answer2_indexes:
            ans2_val = cur_ans2_val
            cur_ans2_val += 1
        else:
            ans2_val = 99999999
        answer2_vals.append(ans2_val)

    # create honeypot column
    cur_honeypot_val = 0
    honeypot_vals = []
    honeypot_indexes = set(honeypot_indexes)
    for i in xrange(len(target_fld_vals)):
        if i in honeypot_indexes:
            honeypot_val = cur_honeypot_val
            cur_honeypot_val += 1
        else:
            honeypot_val = 99999999
        honeypot_vals.append(honeypot_val)

    # adding synthetic answer's preds
    answer_preds = [
        ('answer1', '<', 99999999),
        ('answer2', '<', 99999999),
    ]
    honeypot_preds = [
        ('honeypot', '<', 99999999),
    ]

    # adding additional columns, changing answer columns
    df['answer3'] = answer1_vals
    df['answer4'] = answer2_vals
    df['honeypot'] = honeypot_vals
    answer_preds[0] = ('answer3', answer_preds[0][1], answer_preds[0][2])
    answer_preds[1] = ('answer4', answer_preds[1][1], answer_preds[1][2])
    honeypot_preds[0] = ('honeypot', honeypot_preds[0][1], honeypot_preds[0][2])
    preds += answer_preds + honeypot_preds
    meta['answer_preds'] = answer_preds
    meta['answer_pids'] = [len(preds) - 3, len(preds) - 2]
    meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)
    qs = Query.get_pandas_query_from_preds(answer_preds)
    meta['answer_counts'] = target_counts #np.histogram(df.query(qs)[meta['target_fld']], meta['adjusted_bin_edges'])[0]
    meta['target_counts'] = target_counts #meta['answer_counts']

    # updating pred pair meta
    if pred_pair_counts:
        new_pred_ids = [len(preds) - 3, len(preds) - 2, len(preds) - 1]
        update_pred_pair_meta_with_new_preds(new_pred_ids, df, preds, meta, bin_counts,
                                             target_fld_vals, pred_pair_counts, bin_pred_probs,
                                             bin_pred_pair_probs)







###############################################################################
# experiment running code
###############################################################################

def gen_v7_syn_answer(np_random, original_df, answer_preds, answer_corr, num_bins=10,
                      target_answer_percent_uniform=0.333, answer_slack=0.0,
                      target_fld='target_fld', print_details=True, min_meta=False,
                      honeypot_percent_answer_used=0.7,
                      honeypot_percent_noise_used=0.3,
                      percent_ans1_noise_to_remove=0.5):
    """
    Generates target answer in df
    """
    df = copy.deepcopy(original_df)
    num_rows = len(df)

    # 1. getting target values and bins; outputting correlations for answer preds
    target_fld_bin_vals, bin_counts, adjusted_bin_edges = get_target_fld_vals(num_rows)
    p1, p2 = answer_preds

    # 2. determining average overlap (as initial new_cc)
    cnt_data = []
    for bid, r in enumerate(bin_counts):
        start = sum(bin_counts[0:bid])
        end = sum(bin_counts[0:bid + 1])
        bin_df = df.loc[start:end - 1, :]
        cc = len(bin_df.query(Query.get_pandas_query_from_preds(answer_preds)))
        c1 = len(bin_df.query(Query.get_pandas_query_from_preds([p1])))
        c2 = len(bin_df.query(Query.get_pandas_query_from_preds([p2])))
        cnt_data.append((r, cc, c1, c2, start, end))
    new_cc = int(round(np.mean([x[1] for x in cnt_data])))

    # 3. finding new overlap (must allow for enough rows to overlap w/ desired probability)
    while True:
        bad_cc = False
        for bid, (r, cc, c1, c2, start, end) in enumerate(cnt_data):
            dep_str = 1.0 + answer_corr  # old way: (r * cc) / float(c1 * c2)
            new_c = int(round(np.sqrt((r * new_cc) / dep_str)))
            if ((2 * (new_c - new_cc)) + new_cc) > r:
                bad_cc = True
                break
            if new_c > r or new_cc > new_c:
                bad_cc = True
                break
        if not bad_cc:
            break
        else:
            new_cc -= 1
        if new_cc <= 0:
            raise Exception("new_cc = {}. exiting".format(new_cc))

    # 4. adjust c1 and c2 to minimize correlation change from changing cc to avg_cc
    target_counts = []
    for bid, (r, cc, c1, c2, start, end) in enumerate(cnt_data):
        # 4a. determining new_c value
        dep_str = 1.0 + answer_corr  # old way: (r * cc) / float(c1 * c2)
        new_c = int(round(np.sqrt((r * new_cc) / dep_str)))

        # 4b. getting indexes for val combos of c1=v1, c1=-1, c2=v2, c2=-1
        bin_df = df.loc[start:end - 1, :]
        val_combo_indexes = {
            'nn': list(bin_df.query('({} == -1) and ({} == -1)'.format(p1[0], p2[0])).index.values),
            'vn': list(bin_df.query('({} == {}) and ({} == -1)'.format(p1[0], p1[2], p2[0])).index.values),
            'nv': list(bin_df.query('({} == -1) and ({} == {})'.format(p1[0], p2[0], p2[2])).index.values),
            'vv': list(bin_df.query('({} == {}) and ({} == {})'.format(p1[0], p1[2], p2[0], p2[2])).index.values),
        }
        for key in val_combo_indexes.keys():
            np_random.shuffle(val_combo_indexes[key])

        # 4c. getting +/- counts from each combo's relation w/ new_c & new_cc
        change_counts = {
            'vv': new_cc - len(val_combo_indexes['vv']),
            'nn': (r - new_cc - (2 * (new_c - new_cc))) - len(val_combo_indexes['nn']),
            'vn': (new_c - new_cc) - len(val_combo_indexes['vn']),
            'nv': (new_c - new_cc) - len(val_combo_indexes['nv']),
        }

        # 4d. determining add and remove indexes
        add_cnt = 0
        rm_cnt = 0
        add_key_counts = {}
        alter_indexes = {}
        for key, cnt in change_counts.iteritems():
            if cnt > 0:
                add_cnt += cnt
                add_key_counts[key] = cnt
            else:
                rm_cnt += len(val_combo_indexes[key][0:abs(cnt)])
                alter_indexes[key] = val_combo_indexes[key][0:abs(cnt)]
        if add_cnt != rm_cnt:
            raise Exception("Error: add_cnt={} != rm_cnt={}".format(add_cnt, rm_cnt))

        # 4e. moving indexes around to match final counts
        add_keys = add_key_counts.keys()
        alter_keys = alter_indexes.keys()
        add_key_index = 0
        alter_key_index = 0
        for i in xrange(add_cnt):
            # find the next valid add_key
            has_valid_add_key = False
            for j in xrange(len(add_keys)):
                add_key = add_keys[add_key_index % len(add_keys)]
                add_key_index += 1
                if add_key_counts[add_key] > 0:
                    has_valid_add_key = True
                    add_key_counts[add_key] -= 1
                    break
            if not has_valid_add_key:
                raise Exception("Error: no valid add_key found!")

            # find the next valid alter_key
            has_valid_alter_key = False
            for j in xrange(len(alter_keys)):
                alter_key = alter_keys[alter_key_index % len(alter_keys)]
                alter_key_index += 1
                if len(alter_indexes[alter_key]) > 0:
                    has_valid_alter_key = True
                    alter_index = alter_indexes[alter_key].pop()
                    break
            if not has_valid_alter_key:
                raise Exception("Error: no valid alter_key found!")

            # switching alter_index's config to add_key config
            if add_key == 'nn' and alter_key in ('vn', 'nv', 'vv'):
                df.loc[alter_index, [p1[0], p2[0]]] = [-1, -1]
            elif add_key == 'vn' and alter_key in ('nn', 'nv', 'vv'):
                df.loc[alter_index, [p1[0], p2[0]]] = [p1[2], -1]
            elif add_key == 'nv' and alter_key in ('nn', 'vn', 'vv'):
                df.loc[alter_index, [p1[0], p2[0]]] = [-1, p2[2]]
            elif add_key == 'vv' and alter_key in ('nn', 'vn', 'nv'):
                df.loc[alter_index, [p1[0], p2[0]]] = [p1[2], p2[2]]
            else:
                raise Exception("Invalid add_key / alter_key combo")

        # 4f. updating target_counts
        bin_df = df.loc[start:end - 1, :]
        actual_cc = len(bin_df.query(Query.get_pandas_query_from_preds(answer_preds)))
        target_counts.append(actual_cc)

    # 5. if answer_slack > 0, update target counts to have +/- some slack
    answer_target_counts = copy.deepcopy(target_counts)
    if answer_slack > 0.0:
        for bid, targ_cnt in enumerate(target_counts):
            slack_sd = answer_slack * targ_cnt * 0.25
            cnt = int(round(np_random.normal(loc=targ_cnt, scale=slack_sd, size=1)[0]))
            if cnt > targ_cnt + (answer_slack * targ_cnt):
                cnt = targ_cnt + (answer_slack * targ_cnt)
            elif cnt < targ_cnt - (answer_slack * targ_cnt):
                cnt = targ_cnt - (answer_slack * targ_cnt)
            answer_target_counts[bid] = cnt

    # 6. adding target answer column to dataframe
    final_target_fld_vals = np.zeros((num_rows, 1))
    answer_df = df.query(Query.get_pandas_query_from_preds(answer_preds))
    answer_indexes = list(answer_df.index.values)
    remaining_indexes = list(set(range(0, num_rows)) - set(answer_indexes))
    np_random.shuffle(answer_indexes)
    remaining_target_fld_vals = []
    for bid, cnt in enumerate(answer_target_counts):
        row_ids = answer_indexes[0:cnt]
        del answer_indexes[0:cnt]
        final_target_fld_vals[row_ids, 0] = target_fld_bin_vals[bid][0:cnt]
        remaining_target_fld_vals += target_fld_bin_vals[bid][cnt:]
    final_target_fld_vals[remaining_indexes, 0] = remaining_target_fld_vals
    df[target_fld] = final_target_fld_vals

    # generating answer & honey pot indexes
    bin_counts = np.histogram(df[target_fld], adjusted_bin_edges)[0]
    honeypot_indexes = []
    for bid, r in enumerate(bin_counts):
        start = sum(bin_counts[0:bid])
        end = sum(bin_counts[0:bid + 1])
        bin_df = df.loc[start:end - 1, :]

        bin_answer_indexes = bin_df.query(Query.get_pandas_query_from_preds(answer_preds)).index.tolist()
        bin_noise_indexes = list(set(range(start, end)) - set(bin_answer_indexes))

        num_honeypot_ans_used = int(round(honeypot_percent_answer_used * len(bin_answer_indexes)))
        num_honeypot_noise_used = int(round(honeypot_percent_noise_used * len(bin_noise_indexes)))
        honeypot_indexes += bin_answer_indexes[0:num_honeypot_ans_used] + bin_noise_indexes[0:num_honeypot_noise_used]

        # reducing some noise from syn answer 1
        ans1_bin_df = bin_df.query(Query.get_pandas_query_from_preds([answer_preds[0]]))
        ans1_noise_indexes = ans1_bin_df.ix[bin_noise_indexes].dropna().index.tolist()
        # print("len(ans1_bin_df)={}, len(ans1_noise_indexes)={}, len(bin)={}, len(ans)={}"
        #       .format(len(ans1_bin_df), len(ans1_noise_indexes), bin_counts[bid], len(bin_answer_indexes)))
        num_to_remove = int(round(percent_ans1_noise_to_remove * len(ans1_noise_indexes)))
        df.loc[ans1_noise_indexes[0:num_to_remove], answer_preds[0][0]] = -1


    # create honeypot column
    cur_honeypot_val = 0
    honeypot_vals = []
    honeypot_indexes = set(honeypot_indexes)
    for i in xrange(len(df)):
        if i in honeypot_indexes:
            honeypot_val = cur_honeypot_val
            cur_honeypot_val += 1
        else:
            honeypot_val = 99999999
        honeypot_vals.append(honeypot_val)

    # returning data
    answer1_vals = df[answer_preds[0][0]].tolist()
    answer2_vals = df[answer_preds[1][0]].tolist()
    return (final_target_fld_vals, answer1_vals, answer2_vals, honeypot_vals,
            target_fld, target_counts, adjusted_bin_edges, bin_counts)



def update_v9_meta_with_v7_data(corr, syn_data_mode, df, meta, preds, random_seed,
                                pred_pair_counts, bin_pred_probs, bin_pred_pair_probs):

    # getting data and meta data
    np_random = np.random.RandomState(random_seed)
    if syn_data_mode == "v7":
        honeypot_percent_answer_used = bounded_normal_draw(np_random, loc=0.3, scale=0.1, min_val=0.1, max_val=0.9)
        honeypot_percent_noise_used = 0.6
        percent_ans1_noise_to_remove = 0.5
    elif syn_data_mode == "v7b":
        honeypot_percent_answer_used = bounded_normal_draw(np_random, loc=0.3, scale=0.1, min_val=0.1, max_val=0.9)
        honeypot_percent_noise_used = bounded_normal_draw(np_random, loc=0.3, scale=0.1, min_val=0.1, max_val=0.9)
        percent_ans1_noise_to_remove = bounded_normal_draw(np_random, loc=0.2, scale=0.1, min_val=0.1, max_val=0.9)
    elif syn_data_mode == "v7c":
        honeypot_percent_answer_used = 0.7
        honeypot_percent_noise_used = 0.3
        percent_ans1_noise_to_remove = 0.1

    # grab two random columns, switch to binary, use as answer_preds
    answer_cols = df.columns[4:6].tolist()
    answer_preds = []
    for answer_col in answer_cols:
        df.loc[df[answer_col] >= 0, answer_col] = 1
        df.loc[df[answer_col] < 0, answer_col] = -1
        answer_preds.append((answer_col, '==', 1))

    # generating v7 answer
    (target_fld_vals, answer1_vals, answer2_vals, honeypot_vals,
                target_fld, target_counts, adjusted_bin_edges, bin_counts) = \
        gen_v7_syn_answer(np_random, df, answer_preds, corr,
                          honeypot_percent_answer_used=honeypot_percent_answer_used,
                          honeypot_percent_noise_used=honeypot_percent_noise_used,
                          percent_ans1_noise_to_remove=percent_ans1_noise_to_remove)
    meta['target_fld'] = target_fld
    meta['target_counts'] = target_counts
    meta['adjusted_bin_edges'] = adjusted_bin_edges
    meta['bin_counts'] = bin_counts

    # adding additional columns, changing answer columns
    df['answer3'] = answer1_vals
    df['answer4'] = answer2_vals
    df['honeypot'] = honeypot_vals
    answer_preds[0] = ('answer3', answer_preds[0][1], answer_preds[0][2])
    answer_preds[1] = ('answer4', answer_preds[1][1], answer_preds[1][2])
    honeypot_preds = [('honeypot', '<', 99999999)]
    preds[0] = answer_preds[0]
    preds[1] = answer_preds[1]
    preds[2] = honeypot_preds[0]
    meta['answer_preds'] = answer_preds
    meta['answer_pids'] = [0, 1]
    meta['answer_query'] = Query.get_pandas_query_from_preds(answer_preds)
    qs = Query.get_pandas_query_from_preds(answer_preds)
    meta['answer_counts'] = target_counts #np.histogram(df.query(qs)[meta['target_fld']], meta['adjusted_bin_edges'])[0]
    meta['target_counts'] = target_counts #meta['answer_counts']

    # updating pred pair meta
    if pred_pair_counts:
        new_pred_ids = [0, 1, 2]
        update_pred_pair_meta_with_new_preds(new_pred_ids, df, preds, meta, bin_counts,
                                             target_fld_vals, pred_pair_counts, bin_pred_probs,
                                             bin_pred_pair_probs)







###############################################################################
# experiment running code
###############################################################################

def get_data_cache_key(syn_data_mode, num_rows, num_columns, num_preds, corr, random_seed):
    if syn_data_mode in ("v1", "v1b", "v1c"):
        data_cache_key = 'r{}-c{}-corr{}-seed{}'.format(num_rows, num_columns, corr, random_seed)
    elif syn_data_mode in ("v2", "v2b", "v2c"):
        if num_preds is None:
            raise Exception("Need num_preds set!")
        data_cache_key = 'r{}-c{}-p{}-seed{}'.format(num_rows, num_columns, num_preds, random_seed)
    elif syn_data_mode in ("v7", "v7b", "v7c", "v8", "v8b", "v8c", "v9", "v9b", "v9c"):
        if num_preds is None:
            raise Exception("Need num_preds set!")
        data_cache_key = 'r{}-c{}-p{}-corr{}-seed{}'.format(num_rows, num_columns, num_preds, corr, random_seed)
    else:
        raise Exception("Invalid syn_data_mode = {}".format(syn_data_mode))
    return data_cache_key


def test_syn_data_v2(num_rows, num_columns, corr, random_seed, pred_dep_percent,
                     print_details=False, mip_solver_timeout=600,
                     dep_scoring_method=None, dep_sorting_method=None,
                     min_meta=True, ignore_meta_cache=False, sorting_random_seed=None,
                     slack_percent=0.2, minos_options=None, system="emerilRandom",
                     max_preds=2, syn_data_mode="v1", num_preds=None,
                     min_print_details=True, solver="minos",
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

    data_cache_key = get_data_cache_key(syn_data_mode, num_rows, num_columns, num_preds, corr, random_seed)
    sorting_random_seed = sorting_random_seed if sorting_random_seed else random_seed
    meta_cache_key = data_cache_key + '-sys.{}-pdp{}-sort{}-sp{}'\
                        .format(cur_sys, pred_dep_percent, sorting_random_seed, slack_percent)
    if add_timeout_to_meta_name:
        meta_cache_key += '-time{}'.format(mip_solver_timeout)
    if use_code_sim:
        meta_cache_key += '-codesim'
    meta_cache_file = os.path.join(DATA_GEN_EXP_META_DIR, syn_data_mode, meta_cache_key + ".cache")
    if not ignore_meta_cache and os.path.exists(meta_cache_file):
        with open(meta_cache_file) as f:
            meta = pickle.load(f)
    else:
        # 1. getting data
        df, preds, meta, pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs = \
                get_syn_data_and_pred_pair_meta(num_rows, num_columns, num_preds, corr,
                                                random_seed, system, syn_data_mode,
                                                print_details=print_details,
                                                min_print_details=min_print_details,
                                                use_code_sim=use_code_sim)
        if 'user_pred_ids' not in meta:
            meta['user_pred_ids'] = None

        # 2. get dep scores for experiment params (dep score func & sorting, pred_dep_percent)
        if dep_sorting_method is not None:
            start = timer()
            np_random = np.random.RandomState(sorting_random_seed)
            dep_scores_pdp = 1.0 if system in ("emerilThresRand", "emerilThresRandIndep", "emerilTriHybrid", "emerilTiHybrid", "emerilThresIndep") else pred_dep_percent
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
        if system in ("emerilThresRand", "emerilThresRandIndep", "emerilTriHybrid", "emerilTiHybrid", "emerilThresIndep"):
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
                                               syn_data_mode=syn_data_mode))
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
                                             syn_data_mode=syn_data_mode,
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
                                          syn_data_mode=syn_data_mode,
                                          minos_options=minos_options,
                                          min_print_details=min_print_details,
                                          solver=solver,
                                          user_pred_ids=meta['user_pred_ids']))
        if min_print_details or print_details:
            print("RUNTIME: solving problem with cur system: {:.3f} sec".format(timer() - start))

        # 5. saving meta cache
        with open(meta_cache_file, "w") as f:
            pickle.dump(meta, f, -1)

    if print_details:
        target_bounds = get_target_count_bounds(meta['target_counts'], slack_percent)
        print "--------------------------"
        print "target bounds: {}".format(target_bounds)
        print "answer counts: {}".format(meta['answer_counts'])
        print "answer pids: {}".format(meta['answer_pids'])

    if min_print_details or print_details:
        print "RESULT: found={}, valid={}, final_answer_pids={}"\
            .format(meta['solution_found'], meta['is_valid_solution'], meta['final_answer_pids'])

    return meta
