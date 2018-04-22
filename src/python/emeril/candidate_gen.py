from . import *

###############################################################################
########################### Candidate Gen #####################################

def get_target_bin_counts(bins, count_field='count'):
    adjusted_total_rows = None
    for bin_id, b in enumerate(bins):
        cur_adjusted_total = int(math.floor(b[count_field] / b['target_percent']))
        if adjusted_total_rows is None or cur_adjusted_total < adjusted_total_rows:
            adjusted_total_rows = cur_adjusted_total
    target_bin_counts = {}
    for bin_id, b in enumerate(bins):
        target_bin_counts[bin_id] = int(math.floor(b['target_percent'] * adjusted_total_rows))
    return adjusted_total_rows, target_bin_counts


def get_bins_with_target_percent(df, db_meta, target_field, target_signal):
    field_type = db_meta.get_field_type(target_field)
    bin_edges = db_meta.get_field_bins(target_field)
    bin_counts, _ = np.histogram(df[target_field], bins=bin_edges)  # all but last half-open

    bins = []
    tp_divisor = np.trapz(target_signal)
    tp_multiplier = len(target_signal) / float(len(bin_counts))
    for i in range(len(bin_counts)):
        # determining percentage under target_signal curve current bin represents
        tp_start = int(math.ceil(i * tp_multiplier))
        tp_end = int(math.ceil((i + 1) * tp_multiplier))
        if field_type == 'numeric':
            target_percent = np.trapz(target_signal[tp_start:tp_end]) / tp_divisor
        elif field_type == 'categorical':
            target_percent = target_signal[i]
        else:
            raise Exception("Invalid field type")

        bins.append({
            'target_percent': target_percent,
            'count': bin_counts[i],
            'min_closed': bin_edges[i],
            'max_open': bin_edges[i+1] if i < len(bin_counts) - 1 else bin_edges[i+1] + 1,
        })
    return bins


def get_candidate_preds_v3(df, db_meta, constraints, cache_key,
                           use_min_density=False, use_sub_bin_count=False):
    """
    returns candidate predicates for each want/no_want field

    TODO: This is target depdendent; will need revising when supporting
    multiple targets.
    """
    print("\n====== Starting candidate generation =========")
    if not use_min_density and not use_sub_bin_count:
        raise Exception("must specify use_min_density or use_sub_bin_count")
    elif use_min_density and use_sub_bin_count:
        raise Exception("can't specify both use_min_density and use_sub_bin_count")

    ## 1. loading from cache if exists ##
    if use_min_density:
        final_cache_key = cache_key + '.use_min_density'
    elif use_sub_bin_count:
        final_cache_key = cache_key + '.use_sub_bin_count'
    else:
        raise Exception("Unsupported mode")
    if os.path.exists(os.path.join(CANDIDATE_PREDS_PICKLE_PATH, final_cache_key)):
        with open(os.path.join(CANDIDATE_PREDS_PICKLE_PATH, final_cache_key), 'rb') as f:
            candidate_data = pickle.load(f)
            print("Cache of candidates found; dense_regions={}, "
                  "other_cands={}, on_pred_cands={}, no_pred_other_cands={}"
                  .format(sum([len(preds) for preds in candidate_data['dense_regions'].values()]),
                          len(candidate_data['other_candidates']),
                          len(candidate_data['no_pred_candidates']),
                          len(candidate_data['no_pred_other_candidates'])))
            return candidate_data

    ## 2. getting target signal and field ##
    # TODO(Dolan): add support for more than one target
    print("\n2. Loading target field/signal (TODO: fix for more than one target)...")
    if len(constraints['want_change_signals']) > 1:
        raise Exception("UNSUPPORTED: only supports one want_change_signals")
    target_field, target_signal = constraints['want_change_signals'][0]

    ## 3. Split result into bins and find distrib percent for each ##
    print("\n3. Split result into bins and find distrib percent for each...")
    bins = get_bins_with_target_percent(df, db_meta, target_field, target_signal)
    counts = [x['count'] for x in bins]
    percents = [x['target_percent'] for x in bins]
    print("bin counts: {}".format(', '.join(map(str, counts))))
    print("target_percents: {}".format(', '.join(['{:.2f}'.format(x) for x in percents])))

    ## 4. Determine target bin counts based on target bin percentages and available rows
    ##    Example: If bin5={3142 records, 21% area}, then adjusted_total_rows=16536;
    ##             but if bin3={1148, 12% area}, then adjusted_total_rows=9566;
    ##             repeat this until lowest value found.
    print("\n4. Determine target bin counts based on target bin percentages and available rows...")
    adjusted_total_rows, target_bin_counts = get_target_bin_counts(bins)
    print("total_rows={}".format(len(df)))
    print("adjusted_total_rows={}".format(adjusted_total_rows))
    print target_bin_counts

    ## 5. Checking if dataset can satisfy desired histogram ##
    # first, order histogram percentages and dataset counts, looking for conflict
    print("\n5. Checking if dataset can satisfy desired histogram...")
    sorted_count_indexes = np.argsort(counts)[::-1]
    sorted_percent_indexes = np.argsort(percents)[::-1]
    kt_corr, kt_pval = kendalltau(counts, percents)
    print("kendall tau of counts vs. percents: {:.3f} (pval={:.2f})".format(kt_corr, kt_pval))
    if adjusted_total_rows > 0:
        print("adjusted_total_rows > 0, so safe to proceed.")
    else:
        print("EXITING: adjusted_total_rows == 0")
        exit()

    ## 6. For each bin, find all dense regions in attribute distribs ##
    print("\n6. For each bin, find all dense regions in attribute distribs...")
    dense_regions = defaultdict(list)
    searchable_fields = db_meta.get_searchable_fields()
    for bin_id, b in enumerate(bins):
        target_removal_count = b['count'] - target_bin_counts[bin_id]
        bin_rm_goal_pct = (float(target_removal_count) / b['count'])

        bin_df = df[(df[target_field] >= b['min_closed']) & (df[target_field] < b['max_open'])]
        for fld in searchable_fields:
            # 6a. get current attrib data
            sub_bin_df = bin_df[map(lambda x: np.isreal(x) and x >= 0.0, bin_df[fld])][fld]
            tot_sub_bin_count = len(sub_bin_df)
            if not tot_sub_bin_count:
                # print("NO ROWS FOR ATTRIB: {}".format(fld))
                continue

            # 6b. get histogram for current attrib on current bin
            field_type = db_meta.get_field_type(fld)
            # TODO: for more granularity, use bin-edges from sub_bin
            sub_bin_edges = db_meta.field_bins[fld]
            sub_bin_counts, _ = np.histogram(sub_bin_df, bins=sub_bin_edges)  # all but last half-open

            # 6c. find dense regions (split into bins, note regions > 10%)
            sub_bin_candidates = []
            for i in range(len(sub_bin_counts)):
                min_closed = sub_bin_edges[i]
                max_open = sub_bin_edges[i+1] if i < len(sub_bin_counts) - 1 else sub_bin_edges[i+1] + 1
                sub_bin_percent = float(sub_bin_counts[i]) / tot_sub_bin_count
                bin_percent = float(sub_bin_counts[i]) / b['count']
                region = (fld, sub_bin_counts[i], bin_percent, min_closed, max_open)
                sub_bin_candidates.append((sub_bin_percent, region))
            if use_min_density:
                for sub_bin_percent, region in sub_bin_candidates:
                    if sub_bin_percent >= MIN_DENSITY:
                        dense_regions[bin_id].append(region)
            elif use_sub_bin_count:
                sub_bin_candidates.sort(reverse=True, key=lambda x: x[0])
                for sub_bin_percent, region in sub_bin_candidates[0:TOP_SUB_BIN_COUNT]:
                    # only adding if > 0 percent removed
                    if sub_bin_percent > 0.0:
                        dense_regions[bin_id].append(region)
            else:
                raise Exception("unsupported mode")

        print("bin={}, count_goal={}, removal_goal={} ({:.2f}%), regions={}"\
              .format(bin_id, target_bin_counts[bin_id],
                      target_removal_count, 100.0*bin_rm_goal_pct,
                      len(dense_regions[bin_id])))


    # 8. removing all preds with field = target_field or in no_pred_fields
    no_pred_candidates = defaultdict(list)
    print("\n8. Removing all preds with field = target_field...")
    for bin_id in dense_regions.keys():
        to_remove = []
        for pred_id, (fld, sub_bin_count, bin_percent, min_closed, max_open) in enumerate(dense_regions[bin_id]):
            if fld == target_field:
                if fld in constraints['no_pred_fields']:
                    no_pred_candidates[bin_id].append(dense_regions[bin_id][pred_id])
                to_remove.append(pred_id)
        to_remove.sort(reverse=True)
        for pred_id in to_remove:
            del dense_regions[bin_id][pred_id]
        print("bin={}, removing={}".format(bin_id, len(to_remove)))
    total_preds = sum([len(preds) for preds in dense_regions.values()])
    print("Updated total predicates={} (no_pred_candidates={})".format(total_preds, len(no_pred_candidates)))


    # 9. get eq/lte/gte predicates for all edges
    print("\n9. get eq/lte/gte predicates for all fields' edges...")
    other_candidates = []
    for fld in searchable_fields:
        field_type = db_meta.get_field_type(fld)
        field_bins = db_meta.field_bins[fld]
        for i in range(len(field_bins)):
            if field_type == 'categorical' and i < len(field_bins) - 1:
                other_candidates.append((fld, '==', field_bins[i]))
            if field_type == 'numeric':
                other_candidates.append((fld, '<=', field_bins[i]))
                other_candidates.append((fld, '>=', field_bins[i]))
            # TODO: other preds:
            # 1. add < & > for numeric
            # 2. add n-choose-k for categorical
            # 3. add multi-bin ranges
            # TODO: try to assign to bins for in-out-ratioing?
            # e.g., could get bin_percent for each, etc.
            # add null filters on each attribute
    print("Created {} other_candidates".format(len(other_candidates)))


    # 10. removing preds w/ field = target_field or in no_preds list
    print("10. removing preds w/ field = target_field or in no_preds list...")
    no_pred_other_candidates = []
    to_remove = []
    for i, (fld, fld_op, val) in enumerate(other_candidates):
        if fld == target_field or fld in constraints['no_pred_fields']:
            if fld in constraints['no_pred_fields']:
                no_pred_other_candidates.append(other_candidates[pred_id])
            to_remove.append(pred_id)
    to_remove.sort(reverse=True)
    for pred_id in to_remove:
        del other_candidates[pred_id]
    print("removed {} other_candidates; {} are no_pred".format(len(to_remove), len(no_pred_other_candidates)))

    candidate_data = {
        'dense_regions': dense_regions,
        'no_pred_candidates': no_pred_candidates,
        'other_candidates': other_candidates,
        'no_pred_other_candidates': no_pred_other_candidates,
    }

    ## caching and then returning ##
    with open(os.path.join(CANDIDATE_PREDS_PICKLE_PATH, final_cache_key), 'wb') as f:
        pickle.dump(candidate_data, f, -1)
    return candidate_data


def get_candidate_preds_v4(df, db_meta, constraints, cache_key,
                           use_min_density=False, use_sub_bin_count=False):
    """
    returns candidate predicates for each want/no_want field

    TODO: This is target depdendent; will need revising when supporting
    multiple targets.
    """
    print("\n====== Starting candidate generation =========")
    if not use_min_density and not use_sub_bin_count:
        raise Exception("must specify use_min_density or use_sub_bin_count")
    elif use_min_density and use_sub_bin_count:
        raise Exception("can't specify both use_min_density and use_sub_bin_count")

    ## 1. loading from cache if exists ##
    if use_min_density:
        final_cache_key = cache_key + '.use_min_density'
    elif use_sub_bin_count:
        final_cache_key = cache_key + '.use_sub_bin_count'
    else:
        raise Exception("Unsupported mode")
    if os.path.exists(os.path.join(CANDIDATE_PREDS_PICKLE_PATH, final_cache_key)):
        with open(os.path.join(CANDIDATE_PREDS_PICKLE_PATH, final_cache_key), 'rb') as f:
            candidate_data = pickle.load(f)
            print("Cache of candidates found; dense_regions={}, "
                  "other_cands={}, on_pred_cands={}, no_pred_other_cands={}"
                  .format(sum([len(preds) for preds in candidate_data['dense_regions'].values()]),
                          len(candidate_data['other_candidates']),
                          len(candidate_data['no_pred_candidates']),
                          len(candidate_data['no_pred_other_candidates'])))
            return candidate_data

    ## 2. getting target signal and field ##
    # TODO(Dolan): add support for more than one target
    print("\n2. Loading target field/signal (TODO: fix for more than one target)...")
    if len(constraints['want_change_signals']) > 1:
        raise Exception("UNSUPPORTED: only supports one want_change_signals")
    target_field, target_signal = constraints['want_change_signals'][0]

    ## 3. Split result into bins and find distrib percent for each ##
    print("\n3. Split result into bins and find distrib percent for each...")
    bins = get_bins_with_target_percent(df, db_meta, target_field, target_signal)
    counts = [x['count'] for x in bins]
    percents = [x['target_percent'] for x in bins]
    print("bin counts: {}".format(', '.join(map(str, counts))))
    print("target_percents: {}".format(', '.join(['{:.2f}'.format(x) for x in percents])))

    ## 4. Determine target bin counts based on target bin percentages and available rows
    ##    Example: If bin5={3142 records, 21% area}, then adjusted_total_rows=16536;
    ##             but if bin3={1148, 12% area}, then adjusted_total_rows=9566;
    ##             repeat this until lowest value found.
    print("\n4. Determine target bin counts based on target bin percentages and available rows...")
    adjusted_total_rows, target_bin_counts = get_target_bin_counts(bins)
    print("total_rows={}".format(len(df)))
    print("adjusted_total_rows={}".format(adjusted_total_rows))
    print target_bin_counts

    ## 5. Checking if dataset can satisfy desired histogram ##
    # first, order histogram percentages and dataset counts, looking for conflict
    print("\n5. Checking if dataset can satisfy desired histogram...")
    sorted_count_indexes = np.argsort(counts)[::-1]
    sorted_percent_indexes = np.argsort(percents)[::-1]
    kt_corr, kt_pval = kendalltau(counts, percents)
    print("kendall tau of counts vs. percents: {:.3f} (pval={:.2f})".format(kt_corr, kt_pval))
    if adjusted_total_rows > 0:
        print("adjusted_total_rows > 0, so safe to proceed.")
    else:
        print("EXITING: adjusted_total_rows == 0")
        exit()




    ## 6. For each bin, find all dense regions in attribute distribs ##
    print("\n6. For each bin, find all dense regions in attribute distribs...")
    dense_regions = defaultdict(list)
    searchable_fields = db_meta.get_searchable_fields()
    for bin_id, b in enumerate(bins):
        target_removal_count = b['count'] - target_bin_counts[bin_id]
        bin_rm_goal_pct = (float(target_removal_count) / b['count'])

        bin_df = df[(df[target_field] >= b['min_closed']) & (df[target_field] < b['max_open'])]
        for fld in searchable_fields:
            # 6a. get current attrib data
            sub_bin_df = bin_df[map(lambda x: np.isreal(x) and x >= 0.0, bin_df[fld])][fld]
            tot_sub_bin_count = len(sub_bin_df)
            if not tot_sub_bin_count:
                # print("NO ROWS FOR ATTRIB: {}".format(fld))
                continue

            # 6b. get histogram for current attrib on current bin
            field_type = db_meta.get_field_type(fld)
            # TODO: for more granularity, use bin-edges from sub_bin
            sub_bin_edges = db_meta.field_bins[fld]
            sub_bin_counts, _ = np.histogram(sub_bin_df, bins=sub_bin_edges)  # all but last half-open

            # 6c. find dense regions (split into bins, note regions > 10%)
            sub_bin_candidates = []
            for i in range(len(sub_bin_counts)):
                min_closed = sub_bin_edges[i]
                max_open = sub_bin_edges[i+1] if i < len(sub_bin_counts) - 1 else sub_bin_edges[i+1] + 1
                sub_bin_percent = float(sub_bin_counts[i]) / tot_sub_bin_count
                bin_percent = float(sub_bin_counts[i]) / b['count']
                region = (fld, sub_bin_counts[i], bin_percent, min_closed, max_open)
                sub_bin_candidates.append((sub_bin_percent, region))
            if use_min_density:
                for sub_bin_percent, region in sub_bin_candidates:
                    if sub_bin_percent >= MIN_DENSITY:
                        dense_regions[bin_id].append(region)
            elif use_sub_bin_count:
                sub_bin_candidates.sort(reverse=True, key=lambda x: x[0])
                for sub_bin_percent, region in sub_bin_candidates[0:TOP_SUB_BIN_COUNT]:
                    # only adding if > 0 percent removed
                    if sub_bin_percent > 0.0:
                        dense_regions[bin_id].append(region)
            else:
                raise Exception("unsupported mode")

        print("bin={}, count_goal={}, removal_goal={} ({:.2f}%), regions={}"\
              .format(bin_id, target_bin_counts[bin_id],
                      target_removal_count, 100.0*bin_rm_goal_pct,
                      len(dense_regions[bin_id])))


    # 8. removing all preds with field = target_field or in no_pred_fields
    no_pred_candidates = defaultdict(list)
    print("\n8. Removing all preds with field = target_field...")
    for bin_id in dense_regions.keys():
        to_remove = []
        for pred_id, (fld, sub_bin_count, bin_percent, min_closed, max_open) in enumerate(dense_regions[bin_id]):
            if fld == target_field:
                if fld in constraints['no_pred_fields']:
                    no_pred_candidates[bin_id].append(dense_regions[bin_id][pred_id])
                to_remove.append(pred_id)
        to_remove.sort(reverse=True)
        for pred_id in to_remove:
            del dense_regions[bin_id][pred_id]
        print("bin={}, removing={}".format(bin_id, len(to_remove)))
    total_preds = sum([len(preds) for preds in dense_regions.values()])
    print("Updated total predicates={} (no_pred_candidates={})".format(total_preds, len(no_pred_candidates)))


    # 9. get eq/lte/gte predicates for all edges
    print("\n9. get eq/lte/gte predicates for all fields' edges...")
    other_candidates = []
    for fld in searchable_fields:
        field_type = db_meta.get_field_type(fld)
        field_bins = db_meta.field_bins[fld]
        for i in range(len(field_bins)):
            if field_type == 'categorical' and i < len(field_bins) - 1:
                other_candidates.append((fld, '=', field_bins[i]))
            if field_type == 'numeric':
                other_candidates.append((fld, '<=', field_bins[i]))
                other_candidates.append((fld, '>=', field_bins[i]))
            # TODO: other preds:
            # 1. add < & > for numeric
            # 2. add n-choose-k for categorical
            # 3. add multi-bin ranges
            # TODO: try to assign to bins for in-out-ratioing?
            # e.g., could get bin_percent for each, etc.
            # add null filters on each attribute
    print("Created {} other_candidates".format(len(other_candidates)))


    # 10. removing preds w/ field = target_field or in no_preds list
    print("10. removing preds w/ field = target_field or in no_preds list...")
    no_pred_other_candidates = []
    to_remove = []
    for i, (fld, fld_op, val) in enumerate(other_candidates):
        if fld == target_field or fld in constraints['no_pred_fields']:
            if fld in constraints['no_pred_fields']:
                no_pred_other_candidates.append(other_candidates[pred_id])
            to_remove.append(pred_id)
    to_remove.sort(reverse=True)
    for pred_id in to_remove:
        del other_candidates[pred_id]
    print("removed {} other_candidates; {} are no_pred".format(len(to_remove), len(no_pred_other_candidates)))

    candidate_data = {
        'dense_regions': dense_regions,
        'no_pred_candidates': no_pred_candidates,
        'other_candidates': other_candidates,
        'no_pred_other_candidates': no_pred_other_candidates,
    }

    ## caching and then returning ##
    with open(os.path.join(CANDIDATE_PREDS_PICKLE_PATH, final_cache_key), 'wb') as f:
        pickle.dump(candidate_data, f, -1)
    return candidate_data


def generate_preds_v5(df, db_meta, target_fld):
    tbl = db_meta.numeric_fields.keys()[0]
    preds = []

    # adding numeric fields' preds (bin ranges + lt/gt each bin edge)
    for fld in db_meta.numeric_fields[tbl]:
        # skipping target field
        if fld == target_fld:
            continue
        for i, bin_start in enumerate(db_meta.field_bins[fld]):
            if i > 0:
                preds.append((fld, '<=', bin_start))
            if i < len(db_meta.field_bins[fld]) - 1:
                bin_end = db_meta.field_bins[fld][i + 1]
                preds.append((fld, 'range', (bin_start, bin_end)))
                preds.append((fld, '>=', bin_start))

    # adding categorical fields' preds
    for fld in db_meta.categorical_fields[tbl]:
        # skipping target field
        if fld == target_fld:
            continue
        unique_vals = db_meta.field_bins[fld][0:-1]  # added extra for ranges, so can leave this out
        for unique_val in unique_vals:
            preds.append((fld, '==', unique_val))

    return preds
