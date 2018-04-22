from . import *

###############################################################################
############################## Misc. Utils ####################################

class SignalGenerationError(Exception):
    pass


class EmptyDataFrameError(Exception):
    pass


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


class Query:
    """
    stores attributes from a query
    """
    tables = None
    predicates = None
    inverted_predicates = False

    def __init__(self, inverted_predicates=False):
        self.tables = []
        self.predicates = []
        self.inverted_predicates = inverted_predicates

    def __hash__(self):
        return hash((tuple(sorted(self.tables)), tuple(sorted(self.predicates))))

    def __str__(self):
        ret = ''
        if self.tables:
            ret += "tables=({}), ".format(', '.join(self.tables))
        ret += "query={}".format(self.get_pandas_query())

    @staticmethod
    def get_qs_pred(fld, fld_operator, value, negation=False):
        if fld_operator == 'range':
            pred = '({} >= {} and {} < {})'.format(fld, value[0], fld, value[1])
        elif type(value) in (float, int, np.float64, np.int64):
            pred = '({} {} {})'.format(fld, fld_operator, value)
        else:
            raise Exception("Non-numieric types not supported")
            pred = '({} {} "{}")'.format(fld, fld_operator, value, type(value))
        if negation:
            pred = '~' + pred
        return pred

    @staticmethod
    def get_pandas_query_from_preds(preds, inverted_predicates=False):
        qs = Query(inverted_predicates=inverted_predicates)
        for pred in preds:
            qs.add_predicate(*pred)
        return qs.get_pandas_query()

    def add_table(self, table):
        self.tables.append(table)

    def add_predicate(self, field, pred_operator, pred_value):
        self.predicates.append((field, pred_operator, pred_value))

    def get_pandas_query(self):
        str_preds = []
        for fld, fld_operator, value in self.predicates:
            pred = Query.get_qs_pred(fld, fld_operator, value, negation=self.inverted_predicates)
            str_preds.append(pred)
        join_op = ' and ' #' and ' if self.inverted_predicates else ' or '
        return join_op.join(str_preds)

    def merge_overlapping_preds(self):
        # 1. sort by fld, fld_op, value
        new_predicates = []
        self.predicates.sort(key=lambda x: (x[0], x[1], x[2]))

        # 2. merging field ranges
        fld_ranges = defaultdict(list)
        for (fld, fld_operator, value) in self.predicates:
            if fld_operator != 'range':
                new_predicates.append((fld, fld_operator, value))
            else:
                min_closed, max_open = value
                if fld in fld_ranges:
                    fld_merged = False
                    for i, (existing_min_closed, existing_max_open) in enumerate(fld_ranges[fld]):
                        if min_closed <= existing_max_open:
                            new_range = (existing_min_closed, max_open)
                            fld_ranges[fld][i] = new_range
                            fld_merged = True
                            break
                    if not fld_merged:
                        fld_ranges[fld].append((min_closed, max_open))
                else:
                    fld_ranges[fld].append((min_closed, max_open))

        # 3. adding merged field range preds to new list
        for fld in sorted(fld_ranges.keys()):
            for value in fld_ranges[fld]:
                new_predicates.append((fld, 'range', value))
        self.predicates = new_predicates


def get_query_from_merged_preds(preds):
    clauses = []
    for (fld, min_closed, max_open) in preds:
        clauses.append('~({} >= {} and {} < {})'.format(fld, min_closed, fld, max_open))
    return ' and '.join(clauses)


def strip_tags(html):
    html = re.sub(r'<script.*>.*</script>', '', html)
    html = re.sub(r'<style.*>.*</style>', '', html)
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def scale(signal, top_of_scale=1.0):
    """scales signal between 0 and 1"""
    signal = np.array(signal)
    sMin = min(signal)
    sMax = max(signal)
    signal = (signal - sMin) / (sMax - sMin)
    return signal * top_of_scale


def base_round(x, base=5):
    """From http://stackoverflow.com/a/2272174/318870"""
    return int(base * round(float(x)/base))


def get_fscore(signal_sim, code_sim, beta=1.0):
    """
    fscore between signal & code sim scores.
    higher favors code_sim; code_sim==recall in wiki
    """
    return (1 + beta**2) * ((signal_sim * code_sim) / ((beta**2 * signal_sim) + code_sim))


def get_mape_sim(vals1, vals2):
    mape = np.mean([abs((vals1[i] - vals2[i]) / vals1[i]) for i in range(len(vals1))])
    mape_sim = 0.0 if mape > 1.0 else 1.0 - mape
    return mape_sim


def get_norm_mae_sim(vals1, vals2):
    max1 = float(max(vals1))
    max2 = float(max(vals2))
    norm1 = [x / max1 for x in vals1]
    norm2 = [x / max2 for x in vals2]
    mae = np.mean([abs(norm1[i] - norm2[i]) for i in range(len(norm1))])
    return 1 - mae


def get_kl_divergence(vals1, vals2, fake_normalize_sim=False):
    # avoiding division by zero
    vals1 = np.array(vals1) + REALLY_SMALL_NUMBER
    vals2 = np.array(vals2) + REALLY_SMALL_NUMBER
    # p1 = vals1 / vals1.sum()
    # p2 = vals2 / vals2.sum()
    kl_div = entropy(vals1, vals2)
    if fake_normalize_sim:
        return np.exp(-kl_div)  # for 'normalized' similarity (1 - <this> for distance)
    else:
        return kl_div  # for non-normalized distance


def get_chi_square(vals1, vals2, fake_normalize_sim=False):
    # avoiding division by zero
    vals1 = np.array(vals1) + REALLY_SMALL_NUMBER
    vals2 = np.array(vals2) + REALLY_SMALL_NUMBER
    # p1 = vals1 / vals1.sum()
    # p2 = vals2 / vals2.sum()
    if fake_normalize_sim:
        return np.exp(-chisquare(vals1, vals2)[0])
    else:
        return chisquare(vals1, vals2)[0]


def make_non_uniform(vals):
    is_uniform = True
    cur_vals = copy.deepcopy(vals)  # make a copy
    for x in cur_vals:
        if x != cur_vals[0]:
            is_uniform = False
            break
    if is_uniform:
        if sum(cur_vals) > 2:
            cur_vals[0] += 1
        else:
            cur_vals[0] += REALLY_SMALL_NUMBER
    return cur_vals


def get_pearson_sim(vals1, vals2):
    vals1 = make_non_uniform(vals1)
    vals2 = make_non_uniform(vals2)
    return pearsonr(vals1, vals2)[0]


def dist_plot(data, bins, print_histogram=False):
    if print_histogram:
        print("Histogram: {}".format(np.histogram(data, bins=bins)))
    cur_plot = sns.distplot(data, bins=bins)
    cur_plot.axes.set_xlim(bins[0], bins[-1])
    cur_plot.axes.set_ylim(0.0, 0.6)
    plt.show()


def bna_dist_plot(before_data, after_data, bins, signals_are_binned=False,
                  before_label="before", after_label="after"):
    if signals_are_binned:
        before_bin_counts = before_data
        after_bin_counts = after_data
    else:
        before_bin_counts, _ = np.histogram(before_data, bins=bins)
        after_bin_counts, _ = np.histogram(after_data, bins=bins)

    # sim_matrix = get_sim_matrix_from_bin_edges(bins)
    # emd_score = get_emd_with_sim_matrix(before_bin_counts, after_bin_counts, bins)
    # kl_div_sim = get_kl_divergence(before_bin_counts, after_bin_counts, fake_normalize_sim=True)

    corr = get_pearson_sim(before_bin_counts, after_bin_counts)
    print("{}: {}".format(before_label, before_bin_counts))
    print("{}: {}".format(after_label, after_bin_counts))
    print("CORR = {:.3f}".format(corr))

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    plot1 = sns.distplot(before_data, bins=bins, ax=ax1)
    plot1.axes.set_xlim(bins[0], bins[-1])
    plot1.axes.set_ylim(0.0, 0.6)
    plot2 = sns.distplot(after_data, bins=bins, ax=ax2)
    plot2.axes.set_xlim(bins[0], bins[-1])
    plot2.axes.set_ylim(0.0, 0.6)
    plt.show()


def clean_df_column_names(df):
    cols = {}
    for col in df.columns:
        orig_col = col
        changed = False
        if '.' in col or ': ' in col:
            col = col.replace('.', '__').replace(': ', '__')
            changed = True
        if re.match(r'^\d+$', col):
            col = 'c_{}'.format(col)
            changed = True
        if changed:
            cols[orig_col] = col
    df.rename(columns=cols, inplace=True)


def new_compare_distribs(vals1, vals2, bins, agg_scores, are_same):
    bc1, bins = np.histogram(vals1, bins=bins)
    bc2, _ = np.histogram(vals2, bins=bins)
    bt1 = float(sum(bc1))
    bt2 = float(sum(bc2))
    # bp1 = [x / bt1 if x / bt1 else REALLY_SMALL_NUMBER for x in bc1]
    # bp2 = [x / bt2 if x / bt2 else REALLY_SMALL_NUMBER for x in bc2]
    bp1 = [x / bt1 for x in bc1]
    bp2 = [x / bt2 for x in bc2]

    print("bc1 = {}".format(", ".join(['{}'.format(x) for x in bc1])))
    print("bc2 = {}".format(", ".join(['{}'.format(x) for x in bc2])))
    print("bp1 = {}".format(", ".join(['{:.1f}'.format(x) for x in bp1])))
    print("bp2 = {}".format(", ".join(['{:.1f}'.format(x) for x in bp2])))
    print("bins ({}): {}".format(len(bins), ", ".join(['{}'.format(x) for x in bins])))
    print("-------------")

    scores = (
        ('chi_count_sim', "chi-square (fake_normalize_sim, counts)", 0.5, get_chi_square(bc1, bc2, fake_normalize_sim=True)),
        ('chi_per_sim', "chi-square (fake_normalize_sim, percents)", 0.5, get_chi_square(bp1, bp2, fake_normalize_sim=True)),
        # ('', "chi-square (standard distance, counts)", 0.5, get_chi_square(bc1, bc2)),
        # ('', "chi-square (standard distance, percents)", 0.5, get_chi_square(bp1, bp2)),
        ('kl_count_sim', "KL div (fake_normalize_sim, counts)", 0.5, get_kl_divergence(bc1, bc2, fake_normalize_sim=True)),
        ('kl_per_sim', "KL div (fake_normalize_sim, percents)", 0.5, get_kl_divergence(bp1, bp2, fake_normalize_sim=True)),
        # ('', "KL div (standard distance, counts)", 0.5, get_kl_divergence(bc1, bc2)),
        # ('', "KL div (standard distance, percents)", 0.5, get_kl_divergence(bp1, bp2)),
        ('cosine_count', "cosine sim (counts)", 0.5, 1 - cosine(bc1, bc2)),
        ('cosine_per', "cosine sim (percents)", 0.5, 1 - cosine(bp1, bp2)),
        ('bray_count', "braycurtis sim (counts)", 0.5, 1 - braycurtis(bc1, bc2)),
        ('bray_per', "braycurtis sim (percents)", 0.5, 1 - braycurtis(bp1, bp2)),
        ('pear_count', "Pearson (non uniform, counts)", 0.5, get_pearson_sim(bc1, bc2)),
        ('pear_per', "Pearson (non uniform, percents)", 0.5, get_pearson_sim(bp1, bp2)),
        ('mape_count', "mape_sim (counts)", 0.5, get_mape_sim(bp1, bp2)),
        ('mape_per', "mape_sim (percents)", 0.5, get_mape_sim(bc1, bc2)),
        ('mae_count', "norm_mae_sim (counts)", 0.5, get_norm_mae_sim(bc1, bc2)),
        ('mae_per', "norm_mae_sim (percents)", 0.5, get_norm_mae_sim(bp1, bp2)),
    )
    for key, label, thres, score in scores:
        is_right = (are_same and score >= thres) or (not are_same and score < thres)
        if is_right:
            agg_scores['right'][key] += 1
        else:
            agg_scores['wrong'][key] += 1
        print("{}: {} = {:.3f}".format('RIGHT' if is_right else 'WRONG', label, score))
    print("Current graphs: are_same={}".format(are_same))

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
    plot1 = sns.distplot(vals1, bins=bins, ax=ax1)
    plot1.axes.set_xlim(bins[0], bins[-1])
    plot1 = sns.distplot(vals2, bins=bins, ax=ax2)
    plot1.axes.set_xlim(bins[0], bins[-1])
    plt.show()


def get_safe_pred(pred):
    if "and" in pred:
        safe_pred = re.sub(r"\((\w+) (.*) and (\w+) (.*)\)", r'("\1" \2 and "\3" \4)', pred)
    else:
        safe_pred = re.sub(r"\((\w+) (.*)\)", r'("\1" \2)', pred)
    safe_pred = safe_pred.replace('==', '=')
    return safe_pred


def get_df_bin_pred_counts(df, bin_edges, cur_preds, target_field, invert_preds=True):
    query = Query(inverted_predicates=invert_preds)
    for pred in cur_preds:
        query.add_predicate(*pred)
    qs = query.get_pandas_query()
    print("Qs: {}".format(qs))
    data = df.query(qs)
    print("Len data: {}".format(len(data)))
    if len(data) > 0:
        dist_plot(data[target_field], bin_edges, print_histogram=True)
    else:
        print("ERROR: no data")


def get_pred_parts(pred):
    ret = {}
    m = re.match(r"\((\w+) (.*) and (\w+) (.*)\)", pred)
    if m:
        return (m.group(1), 'range', (m.group(2), m.group(4)))
    m = re.match(r"\(?(\w+) (.*) (.*)\)?", pred)
    if m:
        return (m.group(1), m.group(2), m.group(3))
    raise Exception("bad pred: {}".format(pred))


def get_pred_sim(pred1, pred2, pre_parsed=False):
    if pred1 == pred2:
        return 1.0
    score = 0.0
    if not pre_parsed:
        p1 = get_pred_parts(pred1)
        p2 = get_pred_parts(pred2)
    else:
        p1 = pred1
        p2 = pred2
    if p1[0] == p2[0]:
        score += 0.5
        # first, if both range and any overlap, increase score
        if p1[1] == 'range' and p2[1] == 'range':
            l1, u1 = p1[2]
            l2, u2 = p2[2]
            if ((l1 >= l2 and l1 <= u2) or
                    (u1 >= l2 and u1 <= u2) or
                    (l2 >= l1 and l2 <= u1) or
                    (u2 >= l1 and u2 <= u1)):
                score += 0.25
        # second, if p1 is range and any overlap, increase score
        elif p1[1] == 'range':
            l1, u1 = p1[2]
            if p2[2] >= l1 and p2[2] <= u1:
                score += 0.25
        # third, if p2 is range and any overlap, increase score
        elif p2[1] == 'range':
            l2, u2 = p2[2]
            if p1[2] >= l2 and p1[2] <= u2:
                score += 0.25
    else:
        # TODO: compare distributions of fields?
        pass
    return score


def reset_random_seed():
    np.random.seed(RANDOM_SEED)


def yes_no(bool_var):
    return "Y" if bool_var else "N"


def get_df_from_many_preds(df, answer_preds):
    tmp_df = copy.deepcopy(df)
    PREDS_PER_SUB_QUERY = 20
    for i in xrange(0, len(answer_preds), PREDS_PER_SUB_QUERY):
        tmp_query = Query(inverted_predicates=False)
        for pred in answer_preds[i:i+PREDS_PER_SUB_QUERY]:
            tmp_query.add_predicate(*pred)
        qs = tmp_query.get_pandas_query()
        tmp_df = tmp_df.query(qs)
        if len(tmp_df) == 0:
            break
    return tmp_df


def get_adjusted_bin_edges(bin_edges, adjust_start=True, adjust_end=True):
    adjusted_bin_edges = []
    for bid, bin_start in enumerate(bin_edges[0:-1]):
        bin_end = bin_edges[bid + 1]
        if adjust_start and bid == 0:
            bin_start -= 0.0001
        if adjust_end and bid == len(bin_edges) - 2:
            bin_end += 0.0001
        adjusted_bin_edges.append(bin_start)
        if bid == len(bin_edges) - 2:
            adjusted_bin_edges.append(bin_end)
    return adjusted_bin_edges


def get_signal_distance(s1, s2):
    return np.mean([abs(s1[i] - s2[i]) for i in xrange(len(s1))])


def print_avg_column_corr(df):
    corr_df = df.corr('pearson')
    corr_df = corr_df.where(np.triu(np.ones(corr_df.shape)).astype(np.bool))
    corr_df = corr_df.stack().reset_index()
    corr_df.columns = ['col1', 'col2', 'corr']
    corr_df = corr_df[corr_df.col1 != corr_df.col2]
    corr_df = corr_df.sort_values('corr', ascending=False)
    print "avg: {:.4f}, stdev: {:.4f}".format(corr_df["corr"].mean(), corr_df["corr"].std())


def get_target_count_bounds(target_counts, slack_percent):
    bounds = []
    for bin_id, target_count in enumerate(target_counts):
        lower = target_count - (slack_percent * target_count)
        if lower < 0:
            lower = 0.0
        upper = target_count + (slack_percent * target_count)
        # if upper is exactly 0, add some padding to account for math not being exact
        if upper == 0:
            upper = 0.000001
        bounds.append((lower, upper))
    return bounds


def bounded_normal_draw(np_random, loc, scale, min_val, max_val):
    """loc is the center, scale is the sdev"""
    val = np_random.normal(loc=loc, scale=scale, size=1)[0]
    if val > max_val:
        val = max_val
    elif val < min_val:
        val = min_val
    return val
