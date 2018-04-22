from . import *
from .candidate_gen import get_bins_with_target_percent

###############################################################################
############################### Algos #########################################

def get_constraints_from_prefs(query, no_change_fields, want_change_signals):
    """
    for a query and user's pref of desired & undesired changes, get constraints
    """
    return {
        'tables_used': query.tables,
        'no_change_fields': no_change_fields,
        'want_change_signals': want_change_signals,
        'strict_preds': query.predicates,
        'no_pred_fields': [],
    }


def emeril_algo4_build_queries(db_meta, df, constraints, candidate_data, orig_query):
    """
    Builds list of queries (unscored)
    """
    print("\n====== Starting query building =========")

    # 1. getting target signal and field
    # TODO(Dolan): add support for more than one target
    if len(constraints['want_change_signals']) > 1:
        raise Exception("UNSUPPORTED: only supports one want_change_signals")
    target_field, target_signal = constraints['want_change_signals'][0]

    # 2. get bin counts and in-out ratios
    bins = get_bins_with_target_percent(df, db_meta, target_field, target_signal)
    io_ratios = db_meta.get_field_in_out_ratios(target_field)

    # 3. finding candidates (grab top X predicates by in-out-ratio from each bin)
    candidate_preds = []
    for bin_id, b in enumerate(bins):
        for i in range(NUM_PREDS_PER_BIN):
            # pred_fscore, candidate_id = pred_scores[bin_id][i]
            io_ratio, in_rm_p, out_kept_p, pred = io_ratios[bin_id][i]

            if in_rm_p > 0.9:
                continue

            candidate_preds.append(pred)
            # try:
            #     candidate_preds.append(candidate_data['dense_regions'][bin_id][candidate_id])
            # except IndexError:
            #     raise Exception("Index not found. Try clearing in-out cache")
    print("{} candidate predicates chosen.".format(len(candidate_preds)))

    # 4. creating candidate queries
    candidate_queries = []
    for i in range(1, MAX_PREDS+1):
        for keys in itertools.combinations(range(len(candidate_preds)), i):
            cand_query = Query()
            for table in orig_query.tables:
                cand_query.add_table(table)
            for key in keys:
                fld, fld_operator, value = candidate_preds[key]
                cand_query.add_predicate(fld, fld_operator, value, negation=True)
            cand_query.merge_overlapping_preds()
            candidate_queries.append(cand_query)
    print("{} candidates queries generated.".format(len(candidate_queries)))
    return candidate_queries
