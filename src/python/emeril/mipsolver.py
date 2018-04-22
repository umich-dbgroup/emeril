from . import *
import os
import signal
import subprocess
import multiprocessing as mp
from .baselines import run_greedy_solver_v1, get_tiresias_baseline_data
from .utils import get_target_count_bounds


EMERIL_AMPL_DIR = os.environ.get('EMERIL_AMPL_DIR', "/home/dol/bin/emeril/ampl")
EMERIL_GLPK_DIR = os.environ.get('EMERIL_GLPK_DIR', "/usr/bin")


DYNAMIC_SNOPT_OPTIONS = [
    ## blank options (does well on food pdp=0.001) ##
    {},

    ## solved issue with food pdp=0.01 run0 ##
    {
        'major_feasibility_tolerance': '0.0001',
        'minor_feasibility_tolerance': '0.0001',
        'major_optimality_tolerance': '0.0001',
    },
]


DYNAMIC_MINOS_OPTIONS = [
    ## blank options (does well on random sp=0.02) ##
    {},

    ## something that seemed to work well(?) ##
    {
        'major_iterations': '500',
    },

    ## syndata_v2.slack.nopre sheet ##
    ## (options that did well on non-random sp=0.05) ##
    {
        'feasibility_tolerance': '0.05',
        'major_iterations': '1000',
        'minor_iterations': '75',
        'crash_option': '2',
        'crash_tolerance': '0.0',
        'meminc': '4.42',
        'superbasics_limit': '1000', # default: 50
    },

    ## syndata_v2.slack_mlist2 ##
    ## (options that did well on random sp=0.05) ##
    {
        'feasibility_tolerance': '0.055', # default: 1.0e-06
        'major_iterations': '500',  # default: 50
        'minor_iterations': '80',  # default: 40
        'crash_tolerance': '0.0',  # default: 0.1
        'completion': 'full',  # default: partial
        'weight_on_linear_objective': '2.0',
        'superbasics_limit': '1000', # default: 50
        # # other options to consider:
        # 'meminc': '4.44',
        # 'crash_option': '2',  # default: 3
        # 'penalty_parameter': '1',  # default: 1.0
        # 'row_tolerance': '0.1',  # default: 1.0e-06
        # 'partial_price': '9',  # default: 10
        # 'scale_option': '1',  # default: 2
    },

    ## does well on random sp=0.05 ##
    {
        'crash_tolerance': '0.0',  # default: 0.1
        'major_iterations': '1000',  # default: 50
        'feasibility_tolerance': '0.055',
        'minor_iterations': '80',  # default: 40
        'completion': 'full',  # default: partial
        'weight_on_linear_objective': '2.0',
        'superbasics_limit': '1000', # default: 50
    },

    ## syndata_v2.slack.nopre2 sheet ##
    {
        'feasibility_tolerance': '0.1',
        'major_iterations': '1000',
        'minor_iterations': '500',
        'crash_option': '2',
        'crash_tolerance': '0.0',
        'meminc': '4.42',
        'superbasics_limit': '1000', # default: 50
    },

    ## syndata_v2.slack_mlist2 ##
    {
        'feasibility_tolerance': '0.05',
        'major_iterations': '1000',
        'minor_iterations': '75',
        'meminc': '4.42',
        'superbasics_limit': '1000', # default: 50
    },
]

class SolverFailureSuperbasicsLimit(Exception):
    pass


###############################################################################
############################## MIP Solver #####################################


def get_preds_from_mip_solution(preds, answer_pids, dep_scores=None, print_details=True):
    if print_details:
        print("Loading final answer preds...")
    answer_preds = set()
    final_answer_pids = set()
    for pid in answer_pids:
        if pid < len(preds):
            answer_preds.add(preds[pid])
            final_answer_pids.add(pid)
        elif dep_scores:
            _, pid1, pid2 = dep_scores[pid - len(preds)]
            if print_details:
                print(" - pid={} expanding to {}, {}".format(pid, pid1, pid2))
            answer_preds.add(preds[pid1])
            answer_preds.add(preds[pid2])
            final_answer_pids.add(pid1)
            final_answer_pids.add(pid2)
        else:
            raise Exception("pid out of range, but no dep_scores provided.")
    return list(answer_preds), list(final_answer_pids)


def get_bin_pred_probs(df, pred_tuples, target_fld, adjusted_bin_edges, bin_counts):
    bin_pred_probs = defaultdict(list)
    total_rows = float(len(df))
    for pid, tuple_ids in enumerate(pred_tuples):
        cur_data = df[df.index.isin(tuple_ids)][target_fld]
        bin_pred_counts = np.histogram(cur_data, adjusted_bin_edges)[0]
        bin_pred_probs[pid] = [bin_pred_counts[bid] / float(bin_counts[bid]) for bid in xrange(len(bin_counts))]
    return bin_pred_probs


def get_ampl_result(model_file, mip_solver_timeout, print_details=True, show_output_on_error=True):
    timeout = '/usr/bin/timeout'
    is_mip_timeout = False
    mip_start = timer()
    base_cmd = "ampl {}".format(model_file)
    if mip_solver_timeout is not None:
        cmd = [timeout, str(mip_solver_timeout)] + base_cmd.split(' ')
    else:
        cmd = base_cmd.split(' ')
    if print_details:
        print "COMMAND (with timeout={}, start={}): {}".format(mip_solver_timeout, dt.datetime.now(), base_cmd)
    try:
        mip_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=EMERIL_AMPL_DIR)
    except subprocess.CalledProcessError as e:
        # ignoring timeout issue
        if 'returned non-zero exit status 124' in str(e):
            is_mip_timeout = True
            mip_output = ""
        elif 'Terminating AMPL due to command-line option "-x' in e.output:
            is_mip_timeout = True
            mip_output = e.output
        else:
            print("::: ERROR; OUTPUT BELOW :::")
            print("Exception: {}".format(e))
            print("---------- start of output -----------")
            print(e.output)
            print("---------- end of output -----------")
            raise Exception("Unexpected ampl error occurred.")
    if 'Cannot find "baron"' in mip_output:
        raise Exception("Baron solver not linked up!")
    elif 'Cannot find "minos"' in mip_output:
        raise Exception("MINOS solver not linked up!")
    elif 'Cannot find "snopt"' in mip_output:
        raise Exception("SNOPT solver not linked up!")
    elif 'snopt not licensed for this machine' in mip_output:
        raise Exception("SNOPT license issue!")
    mip_runtime = timer() - mip_start

    # attempting to process output
    mip_answer_ids = None
    m = re.search(r'Chosen pred_ids \(\d+ total\):\s+(.*)\s+------', mip_output, re.M)
    bad_m = re.search(r'Chosen pred_ids \(-\d+ total\):\s+(.*)\s+------', mip_output, re.M)
    if is_mip_timeout:
        if print_details:
            print "WARNING: mip solver timed out."
        mip_status = 'timeout'
    elif "termination code 6" in mip_output:
        if print_details:
            print "WARNING: mip solver had 'termination code 6'."
        mip_status = 'termcode6'
    elif "termination code 11" in mip_output:
        if print_details:
            print "WARNING: mip solver had 'termination code 11'."
        mip_status = 'termcode11'
    elif "infeasible" in mip_output or "Infeasible" in mip_output:
        if print_details:
            print "WARNING: mip solver deemed problem infeasible."
        mip_status = 'infeasible'
    elif "Requested accuracy could not be achieved" in mip_output:
        if print_details:
            print "WARNING: mip solver deemed problem infeasible."
        mip_status = 'unachievable_accuracy'
    elif ("too many major iterations" in mip_output or
          "Major iteration limit reached" in mip_output):
        if print_details:
            print "WARNING: mip solver exceeded major iterations."
        mip_status = 'major_iterations'
    elif "Too many iterations" in mip_output:
        if print_details:
            print "WARNING: mip solver said too many iterations."
        mip_status = 'too_many_iterations'
    elif "Nonlinear infeasibilities minimized" in mip_output:
        if print_details:
            print "WARNING: Nonlinear infeasibilities minimized."
        mip_status = 'infeasibilities_minimized'
    elif "Surprise INFO" in mip_output:
        if print_details:
            print "WARNING: Surprise INFO from snOptB."
        mip_status = 'surprise_info'
    elif "failure: ran out of memory" in mip_output:
        if print_details:
            print "WARNING: ran out of memory."
        mip_status = 'out_of_memory'
    elif "not enough storage for the basis factors" in mip_output:
        if print_details:
            print "WARNING: not enough storage for the basis factors."
        mip_status = 'storage_for_factors'
    elif "Error in basis package" in mip_output:
        if print_details:
            print "WARNING: Error in basis package."
        mip_status = 'error_in_basis'
    elif "unbounded (or badly scaled) problem" in mip_output:
        if print_details:
            print "WARNING: unbounded (or badly scaled) problem."
        mip_status = 'badly_scaled'
    elif "numerical error: the general constraints" in mip_output:
        if print_details:
            print "WARNING: nunumerical error: the general constraints."
        mip_status = 'numerical_error_constraints'
    elif bad_m:
        if print_details:
            print "WARNING: mip solver has negative number of chosen pred_ids."
        mip_status = 'negchosen'
    # new addition (11/28/2017) -- don't care if treated as failure
    elif re.search(r'the superbasics limit \(.*\) is too small.', mip_output):
        print("WARNING: MIP solver error: the superbasics limit is too small.")
        mip_status = 'superbasics_limit_failure'
    elif not m:
        print("::: UNEXPECTED ERROR (regex fail); OUTPUT BELOW :::")
        print(mip_output)
        raise Exception("Unexpected ampl error occurred (regex fail).")
    elif "current point cannot be improved" not in mip_output and \
            "optimal solution found" not in mip_output and \
            "Optimal solution found" not in mip_output and \
            "the objective has not changed for the last" not in mip_output:
        if show_output_on_error:
            print("::: ERROR (optimal solution not found); OUTPUT BELOW :::")
            print(mip_output)
        if re.search(r'the superbasics limit \(.*\) is too small.', mip_output):
            raise SolverFailureSuperbasicsLimit("MIP solver error: the superbasics limit is too small.")
        else:
            raise Exception("Optimal solution not found w/out expected error message.")
    else:
        # processing mip solution, if found
        mip_answer_ids = map(int, [x for x in m.group(1).split(", ") if x.strip()])
        if print_details:
            print("ANSWER PRED_IDS: {}".format(', '.join(map(str, mip_answer_ids))))
        if not mip_answer_ids:
            if print_details:
                print "WARNING: no answer pred_ids from mip solver"
            mip_status = "no answer pids"
        elif "current point cannot be improved" in mip_output:
            if print_details:
                print "WARNING: mip solver says current point not improvable."
            mip_status = 'cannot_be_improved'
        elif "the objective has not changed for the last" in mip_output:
            if print_details:
                print "WARNING: mip solver says not changed in long while."
            mip_status = 'obj_not_changed'
        else:
            mip_status = "success"
    return mip_answer_ids, mip_status, mip_output, mip_runtime


def get_glpk_result(model_file, data_file, mip_solver_timeout, print_details=True):
    """
    Processes ampl model and data file, parsing output
    """
    timeout = '/usr/bin/timeout'

    is_mip_timeout = False
    mip_start = timer()
    base_cmd = "glpsol -m {} -d {}".format(model_file, data_file)
    if mip_solver_timeout is not None:
        cmd = [timeout, str(mip_solver_timeout)] + base_cmd.split(' ')
    else:
        cmd = base_cmd
    if print_details:
        print "COMMAND (with timeout={}, start={}): {}".format(mip_solver_timeout, dt.datetime.now(), base_cmd)
        print
    try:
        mip_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=EMERIL_GLPK_DIR)
    except subprocess.CalledProcessError as e:
        mip_output = ""
        if 'returned non-zero exit status 124' in str(e):
            is_mip_timeout = True
        else:
            print("ERROR !!!! Unknown error occurred:")
            raise e
    mip_runtime = timer() - mip_start

    # attempting to process output
    mip_answer_ids = None
    m = re.search(r'Chosen pred_ids \(\d+ total\):\s+(.*)\s+------', mip_output, re.M)
    if is_mip_timeout:
        if print_details:
            print "WARNING: mip solver timed out."
        mip_status = 'timeout'
    elif "NO PRIMAL FEASIBLE SOLUTION" in mip_output \
            or "NO INTEGER FEASIBLE SOLUTION" in mip_output:
        if print_details:
            print "WARNING: mip solver deemed problem infeasible."
        mip_status = 'infeasible'
    elif not m:
        print("::: UNEXPECTED ERROR (regex fail); OUTPUT BELOW :::")
        print(mip_output)
        raise Exception("Unexpected ampl error occurred (regex fail).")
    else:
        # processing mip solution, if found
        mip_answer_ids = map(int, [x for x in m.group(1).split(", ") if x.strip()])
        if print_details:
            print("ANSWER PRED_IDS: {}".format(', '.join(map(str, mip_answer_ids))))
        if not mip_answer_ids:
            if print_details:
                print "WARNING: no answer pred_ids from mip solver"
            mip_status = "no answer pids"
        elif "the current point cannot be improved" in mip_output:
            if print_details:
                print "WARNING: mip solver says current point not improvable."
            mip_status = 'cannot_be_improved'
        else:
            mip_status = "success"
    return mip_answer_ids, mip_status, mip_output, mip_runtime








###############################################################################
######################### Pred meta and dep scores ############################

def get_utm_coords_from_linear_index(k, n):
    """
    via https://stackoverflow.com/a/27088560/318870
    """
    # i = n - 2 - int(math.floor(math.sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5))
    # j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    try:
        i = int(math.ceil(math.sqrt(2 * k + 0.25) - 0.5))
        j = k - (i - 1) * i / 2
    except:
        raise Exception("Problem with index = {}".format(k))
    return i, j


def _parallel_get_pred_pairs(queue, pairs, pred_counts, pred_tuples, target_fld_vals, adjusted_bin_edges, bin_counts):
    pred_pair_counts = []
    bin_pred_pair_probs = {}
    num_processed = 0
    print("Starting parallel processing ({})".format(dt.datetime.now()))
    for pid1, pid2 in pairs:
        if pid1 == pid2 or pred_counts[pid1] == 0 or pred_counts[pid2] == 0:
            continue
        combo_tuple_ids = list(pred_tuples[pid1].intersection(pred_tuples[pid2]))
        combo_cnt = len(combo_tuple_ids)
        pred_pair_counts.append((pid1, pid2, combo_cnt, pred_counts[pid1], pred_counts[pid2]))
        cur_data = np.take(target_fld_vals, combo_tuple_ids)
        bp_counts = np.histogram(cur_data, adjusted_bin_edges)[0]
        bin_pred_pair_probs[(pid1, pid2)] = [bp_counts[bid] / float(bc) for bid, bc in enumerate(bin_counts)]
        num_processed += 1
    queue.put((pred_pair_counts, bin_pred_pair_probs, num_processed))


def parallel_get_pred_pair_bin_probs(df, preds, target_fld, pred_counts, pred_tuples,
                                     adjusted_bin_edges, bin_counts, n_processes,
                                     print_details=False):
    if print_details:
        print dt.datetime.now(), "- 3. getting bin_pred_pair_probs and pred_pair_counts..."
    bpp_start = timer()
    target_fld_vals = np.array(df[target_fld])

    pairs = itertools.combinations(xrange(len(preds)), 2)
    pair_chunks = [
        itertools.islice(p, i, None, n_processes)
        for i, p in enumerate(itertools.tee(pairs, n_processes))
    ]
    queue = mp.Queue()
    for i, pair_chunk in enumerate(pair_chunks):
        p = mp.Process(target=_parallel_get_pred_pairs,
                       args=(queue, pair_chunk, pred_counts, pred_tuples, target_fld_vals,
                             adjusted_bin_edges, bin_counts))
        p.Daemon = True
        p.start()
    pred_pair_counts = []
    bin_pred_pair_probs = {}
    num_processed = 0
    for i in xrange(len(pair_chunks)):
        result = queue.get()
        pred_pair_counts += result[0]
        bin_pred_pair_probs.update(result[1])
        num_processed += result[2]
    queue.close()

    if print_details:
        print(" - done w/ bpp building ({:.3f} sec, {} processed)".format(timer() - bpp_start, num_processed))
    return pred_pair_counts, bin_pred_pair_probs


def get_pred_pair_bin_probs(df, preds, target_fld, pred_counts, pred_tuples,
                            adjusted_bin_edges, bin_counts, print_details=False):
    bpp_start = timer()
    target_fld_vals = np.array(df[target_fld])
    bin_pred_pair_probs = {}
    pred_pair_counts = []
    num_processed = 0
    for pid1, pid2 in itertools.combinations_with_replacement(xrange(len(preds)), 2):
        if pid1 == pid2 or pred_counts[pid1] == 0 or pred_counts[pid2] == 0:
            continue
        combo_tuple_ids = list(pred_tuples[pid1].intersection(pred_tuples[pid2]))
        combo_cnt = len(combo_tuple_ids)
        pred_pair_counts.append((pid1, pid2, combo_cnt, pred_counts[pid1], pred_counts[pid2]))
        cur_data = np.take(target_fld_vals, combo_tuple_ids)
        bp_counts = np.histogram(cur_data, adjusted_bin_edges)[0]
        bin_pred_pair_probs[(pid1, pid2)] = [bp_counts[bid] / float(bc) for bid, bc in enumerate(bin_counts)]
        num_processed += 1
        if print_details and num_processed % 100000 == 0:
            print " - at {}, time={:.3f}s".format(num_processed, timer() - bpp_start)
    if print_details:
        print(" - done w/ bpp building ({:.3f} sec, {} processed)".format(timer() - bpp_start, num_processed))
    return pred_pair_counts, bin_pred_pair_probs


def get_pred_pair_meta(df, preds, target_fld, adjusted_bin_edges, syn_data_mode,
                       data_cache_key, print_details=False, n_processes=1,
                       skip_loading=False):
    cache_file = os.path.join(PRED_PAIR_META_CACHE_DIR, syn_data_mode, "{}.cache".format(data_cache_key))
    if data_cache_key is not None and os.path.exists(cache_file):
        if skip_loading:
            pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs = [None]*4
        else:
            with open(cache_file) as f:
                pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs = pickle.load(f)
        if print_details:
            print("Pred pair meta cache loaded.")
    else:
        print("WARNING: Building pre pair meta (slow)...")

        # 1. grabbing pred counts/tuples
        start = timer()
        if print_details:
            print dt.datetime.now(), "- 1. grabbing pred counts/tuples..."
        pred_counts = []
        pred_tuples = []
        for pid, pred in enumerate(preds):
            tuple_ids = set(df.query(Query.get_qs_pred(*pred)).index.values)
            pred_counts.append(len(tuple_ids))
            pred_tuples.append(tuple_ids)
        print "done grabbing pred counts/tuples ({:.3f} sec)".format(timer() - start)

        # 2. getting bin_probs and bin_pred_probs
        start = timer()
        if print_details:
            print dt.datetime.now(), "- 2. getting bin_probs and bin_pred_probs..."
        bin_counts = np.histogram(df[target_fld], adjusted_bin_edges)[0]
        total_rows = len(df)
        bin_probs = [x / float(total_rows) for x in bin_counts]
        bin_pred_probs = get_bin_pred_probs(df, pred_tuples, target_fld, adjusted_bin_edges, bin_counts)
        print "done getting bin_probs and bin_pred_probs ({:.3f} sec)".format(timer() - start)

        # 3. getting bin_pred_pair_probs and pred_pair_counts
        start = timer()
        if n_processes > 1:
            pred_pair_counts, bin_pred_pair_probs = \
                    parallel_get_pred_pair_bin_probs(df, preds, target_fld, pred_counts, pred_tuples,
                                                     adjusted_bin_edges, bin_counts, n_processes,
                                                     print_details=print_details)
        else:
            pred_pair_counts, bin_pred_pair_probs = \
                    get_pred_pair_bin_probs(df, preds, target_fld, pred_counts, pred_tuples,
                                            adjusted_bin_edges, bin_counts, print_details=print_details)
        print "done getting bin_pred_pair_probs and pred_pair_counts ({:.3f} sec)".format(timer() - start)

        # 4. saving to disk
        start = timer()
        if print_details:
            print dt.datetime.now(), "- 4. saving to disk..."
        if data_cache_key is not None:
            with open(cache_file, "w") as f:
                pickle.dump((pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs), f, -1)
        print "done saving to disk ({:.3f} sec)".format(timer() - start)

    return pred_pair_counts, bin_probs, bin_pred_probs, bin_pred_pair_probs


def update_pred_pair_meta_with_new_preds(new_pred_ids, df, preds, meta, bin_counts, target_fld_vals,
                                         pred_pair_counts, bin_pred_probs, bin_pred_pair_probs):
    # loading pred counts/tuples for use in updating pred-pair meta
    pred_counts = []
    pred_tuples = []
    for pid, pred in enumerate(preds):
        tuple_ids = set(df.query(Query.get_qs_pred(*pred)).index.values)
        pred_counts.append(len(tuple_ids))
        pred_tuples.append(tuple_ids)

    # updating pred-pair meta (only if pred_pair_counts defined; greedy doesn't use)
    for new_pid in new_pred_ids:
        new_pred = preds[new_pid]

        # updating bin_pred_probs
        tuple_ids = pred_tuples[new_pid]
        cur_data = df[df.index.isin(tuple_ids)][meta['target_fld']]
        bin_pred_counts = np.histogram(cur_data, meta['adjusted_bin_edges'])[0]
        bin_pred_probs[new_pid] = [bin_pred_counts[bid] / float(bin_counts[bid]) for bid in xrange(len(bin_counts))]

        # updating pred_pair_counts and bin_pred_pair_probs
        for pid, pred in enumerate(preds):
            if new_pid == pid:
                continue
            pid1, pid2 = sorted([new_pid, pid])
            combo_tuple_ids = list(pred_tuples[pid1].intersection(pred_tuples[pid2]))
            combo_cnt = len(combo_tuple_ids)
            pred_pair_counts.append((pid1, pid2, combo_cnt, pred_counts[pid1], pred_counts[pid2]))
            cur_data = np.take(target_fld_vals, combo_tuple_ids)
            bp_counts = np.histogram(cur_data, meta['adjusted_bin_edges'])[0]
            bin_pred_pair_probs[(pid1, pid2)] = [bp_counts[bid] / float(bc) for bid, bc in enumerate(bin_counts)]


def get_dep_scores_v2(np_random, pred_pair_counts, total_rows, answer_pids,
                      scoring_method="independence", sorting_method="scores",
                      pred_dep_percent=1.0, print_details=False, min_print_details=True):
    """
    returns dep scores for given method/ranking/percent
    """
    # 1. getting dep_scores
    if print_details:
        print dt.datetime.now(), "1. getting dep_scores..."
    dep_scores = []
    if pred_dep_percent == 0.0:
        print(" - skipping dep scoring since pred_dep_percent = 0.0.")
    elif sorting_method == "random":
        dep_scores = [(0, pid1, pid2) for pid1, pid2, _, _, _ in pred_pair_counts]
        print(" - skipping dep score calculations since sorting = random.")
    else:
        for pid1, pid2, combo_cnt, p1_cnt, p2_cnt in pred_pair_counts:
            pr1 = p1_cnt / float(total_rows)
            pr2 = p2_cnt / float(total_rows)
            pr_combo = combo_cnt / float(total_rows)
            if scoring_method == "independence":
                # if indep, P(p1 & p2) = P(p1)*P(p2)
                # => P(p1 & p2) - P(p1)*P(p2) = 0
                # if dep or anti-dep, abs(P(p1 & p2) - P(p1)*P(p2)) > 0
                score = abs(pr_combo - (pr1 * pr2))
                #score = abs((pr_combo / (pr1 * pr2)) - 1)
            elif scoring_method == "pmi":
                score = math.log((pr_combo + 0.00000000001) / (pr1 * pr2))
            elif scoring_method == "combo_percent":
                score = combo_cnt / float(p1_cnt + p2_cnt)
            elif scoring_method == "abs_count_diff":
                exp_cnt = pr1 * pr2 * total_rows
                score = abs(exp_cnt - combo_cnt)
            else:
                raise Exception("Unsupported scoring_method: {}".format(scoring_method))
            dep_scores.append((score, pid1, pid2))

    # 2. sorting dep_scores
    if print_details:
        print dt.datetime.now(), "2. sorting dep_scores..."
    start = timer()
    if pred_dep_percent == 0.0:
        print(" - skipping dep score sorting since pred_dep_percent = 0.0.")
    else:
        if sorting_method == "scores":
            if scoring_method == "independence":
                is_reverse_sort = True
            elif scoring_method == "combo_percent":
                is_reverse_sort = False
            elif scoring_method == "abs_count_diff":
                is_reverse_sort = True
            else:
                raise Exception("Unsupported scoring_method")
            dep_scores.sort(key=lambda x: x[0], reverse=is_reverse_sort)
        elif sorting_method == "random":
            np_random.shuffle(dep_scores)
        elif sorting_method == 'none':
            pass
        else:
            raise Exception("Invalid sorting argument.")
        if print_details:
            print "Done sorting {} scores ({}s).".format(len(dep_scores), timer() - start)

    # 3. building meta on dep scores
    if print_details:
        print dt.datetime.now(), "3. building meta on dep scores..."
    meta = {}
    meta['num_dep_scores_used'] = int(round(len(pred_pair_counts) * pred_dep_percent))
    meta['num_dep_scores_avail'] = len(pred_pair_counts)
    meta['answer_pids_dep_scores_rank'] = None
    for i, (score, pid1, pid2) in enumerate(dep_scores):
        if (pid1, pid2) == tuple(answer_pids):
            if print_details:
                print "Found answer_pids in dep_scores; at index={} (out of {})".format(i, len(dep_scores))
                print "Num dep scores used: {}".format(meta['num_dep_scores_used'])
            meta['answer_pids_dep_scores_rank'] = i
            break
    if dep_scores and (print_details or min_print_details):
        min_pdp = meta['answer_pids_dep_scores_rank'] / float(meta['num_dep_scores_avail'])
        print("answer dep_info rank={} (min_pdp={:.2f}, used={}, avail={})"
              .format(meta['answer_pids_dep_scores_rank'], min_pdp,
                      meta['num_dep_scores_used'], meta['num_dep_scores_avail']))

    # 4. truncating based on pred_dep_percent
    return dep_scores[0:meta['num_dep_scores_used']], meta










###############################################################################
######################## Emeril AMPL-based MIP Solver #########################

def get_user_pred_sims(pred_ids, preds, user_pred_ids, first_combo_pid, dep_scores):
    user_pred_sims = {}
    for pid in pred_ids:
        cur_pred_sims = []
        for upid in user_pred_ids:
            if pid >= first_combo_pid:
                _, pid1, pid2 = dep_scores[pid - first_combo_pid]
                sim1 = get_pred_sim(preds[upid], preds[pid1], pre_parsed=True)
                sim2 = get_pred_sim(preds[upid], preds[pid2], pre_parsed=True)
                sim = (sim1 + sim2) / 2.0
            else:
                sim = get_pred_sim(preds[upid], preds[pid], pre_parsed=True)
            cur_pred_sims.append(sim)
        user_pred_sims[pid] = cur_pred_sims
    return user_pred_sims


def gen_ampl_data_file_v2(df, preds, dep_scores, bin_probs, bin_pred_probs,
                          bin_pred_pair_probs, target_counts, slack_percent,
                          output_file, user_pred_ids=None):
    """
    Generates data file for input to ampl
    """
    pred_ids = range(len(preds))
    bin_ids = range(len(bin_probs))
    total_rows = len(df)
    if user_pred_ids is None:
        user_pred_ids = []

    # 1. getting pred constraints and combo-pred bin counts
    pred_constraints = []
    constraint_ids = []
    first_combo_pid = max(pred_ids) + 1
    for cid, (_, pid1, pid2) in enumerate(dep_scores):
        combo_pred_id = first_combo_pid + cid
        pred_constraints.append((cid + 1, (pid1, pid2, combo_pred_id)))
        pred_ids.append(combo_pred_id)
        constraint_ids.append(cid + 1)
        bin_pred_probs[combo_pred_id] = bin_pred_pair_probs[(pid1, pid2)]

    # 2. getting user pred sims
    if user_pred_ids:
        user_pred_sims = get_user_pred_sims(pred_ids, preds, user_pred_ids, first_combo_pid, dep_scores)

    # 3. output mathprog
    with open(output_file, "w") as f:
        f.write("set bin_ids := {};\n".format(' '.join(map(str, bin_ids))))
        f.write("set pred_ids := {};\n".format(' '.join(map(str, pred_ids))))
        if user_pred_ids:
            f.write("set user_pred_ids := {};\n".format(' '.join(map(str, user_pred_ids))))
        f.write("set constraint_ids := {};\n".format(' '.join(map(str, constraint_ids))))
        for cid, cur_pred_ids in pred_constraints:
            f.write("set pred_constraints[{}] := {};\n".format(cid, ', '.join(map(str, cur_pred_ids))))
        f.write("\n")

        f.write("param num_bins := {};\n".format(len(bin_ids)))
        f.write("param total_rows := {};\n".format(total_rows))
        f.write("param bin_probs :=\n")
        for bid in bin_ids:
            f.write("  {} {}\n".format(bid, bin_probs[bid]))
        f.write(";\n")
        f.write("param bin_pred_probs: {} :=\n".format(' '.join(map(str, bin_ids))))
        for pid in pred_ids:
            if len(bin_pred_probs[pid]) < len(bin_ids):
                raise Exception("Error: pid={} len(bin_pred_probs)={}".format(pid, len(bin_pred_probs[pid])))
            f.write("  {} {}\n".format(pid, ' '.join(['{:.4f}'.format(x) for x in bin_pred_probs[pid]])))
        f.write(";\n")
        f.write("param target_counts :=\n")
        for bin_id, target_count in enumerate(target_counts):
            f.write("  {} {}\n".format(bin_id, target_count))
        f.write(";\n")
        f.write("param target_bounds: lower upper :=\n")
        target_bounds = get_target_count_bounds(target_counts, slack_percent)
        for bin_id, (lower, upper) in enumerate(target_bounds):
            f.write("  {} {} {}\n".format(bin_id, lower, upper))
        f.write(";\n")
        f.write("\n")

        if user_pred_ids:
            f.write("param user_pred_sims: {} :=\n".format(' '.join(map(str, user_pred_ids))))
            for pid in pred_ids:
                f.write("  {} {}\n".format(pid, ' '.join(['{:.4f}'.format(x) for x in user_pred_sims[pid]])))
            f.write(";\n")
            f.write("\n")

        f.write("param first_combo_pid := {};\n".format(first_combo_pid))
        f.write("end;\n")

    # 7. returning some meta data
    return target_bounds


def run_mip_solver_v2(df, preds, dep_scores, bin_probs, bin_pred_probs,
                      bin_pred_pair_probs, target_counts, target_fld, adjusted_bin_edges,
                      target_answer_pids, slack_percent=0.2, slack_increment=None, slack_max=1.0,
                      mip_solver_timeout=600, print_details=False, filename_uid=None,
                      syn_data_mode="v1", minos_options=None, min_print_details=False,
                      solver='minos', user_pred_ids=None):
    """
    Generates mathprog/ampl file and gets results
    """
    meta = {}

    # 1. determining run params
    # 1a. determining slack percents to use
    slack_percents = [slack_percent]
    if slack_increment:
        if not slack_increment or not slack_max:
            raise Exception("If using slack_max, must use slack_increment (and vice versa).")
        for sp in np.arange(slack_percent + slack_increment, slack_max, slack_increment):
            slack_percents.append(sp)
        if print_details:
            print "Slack percents: {}".format(slack_percents)

    # 1b. determining minos_options
    minos_opt_strs = []
    if minos_options == "dynamic":
        if solver == 'minos':
            all_minos_options = DYNAMIC_MINOS_OPTIONS
        elif solver == 'snopt':
            all_minos_options = DYNAMIC_SNOPT_OPTIONS
        else:
            raise Exception("invalid solver")
    elif minos_options is not None:
        all_minos_options = [minos_options, ]
    else:
        if solver == 'minos':
            all_minos_options = [DYNAMIC_MINOS_OPTIONS[0], ]
        elif solver == 'snopt':
            all_minos_options = [DYNAMIC_MINOS_OPTIONS[0], ]
        else:
            raise Exception("invalid solver")
    for minos_options in all_minos_options:
        minos_opt_str = ' '.join(['{}={}'.format(k, v) for k, v in minos_options.iteritems()])
        minos_opt_strs.append(minos_opt_str)

    # 1c. combining run params
    all_run_params = []
    for sp in slack_percents:
        for minos_opt_str in minos_opt_strs:
            all_run_params.append({'sp': sp, 'minos_opt_str': minos_opt_str})


    # 2. running ampl for each slack percent, stopping once solution found
    meta['solver'] = solver
    meta['solution_found'] = False
    meta['is_valid_solution'] = False
    meta['semi_exact'] = False
    meta['exact_solution'] = False
    meta['solution_sp'] = None
    meta['final_answer_pids'] = None
    meta['mip_runtimes'] = []
    meta['mip_outputs'] = []
    meta['mip_statuses'] = []
    meta['slack_percents'] = []
    meta['minos_opt_strs'] = []
    for run_params in all_run_params:
        sp = run_params['sp']
        minos_opt_str = run_params['minos_opt_str']
        if min_print_details or print_details:
            print "Current slack={}, slack={}, minos_opt_str={}".format(sp, solver, minos_opt_str)

        # 2a. determining model and data file to use
        if filename_uid is None:
            filename_uid = hash(dt.datetime.now())
        data_file = os.path.join(MIP_INPUT_FILE_DIR, syn_data_mode, "{}.slack{}.data".format(filename_uid, sp))
        model_file = os.path.join(MIP_INPUT_FILE_DIR, syn_data_mode, "{}.slack{}.mod".format(filename_uid, sp))
        cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "optimization-models"))
        if user_pred_ids:
            ampl_model_fname = "em-dep-inc-prob-codesim.mod"
        else:
            ampl_model_fname = "em-dep-inc-prob.mod"
        with open(os.path.join(model_dir, ampl_model_fname)) as f:
            model_file_contents = f.read()
        meta['last_model_file'] = model_file

        # 2b. setting minos options
        with open(model_file, "w") as f:
            if solver == 'minos':
                mip_opt_line = "option minos_options '{}';".format(minos_opt_str)
            elif solver == 'snopt':
                mip_opt_line = "option snopt_options '{}';".format(minos_opt_str)
            else:
                raise Exception("invalid solver")
            contents = model_file_contents.replace("{{ MIP_SOLVER }}", solver)
            contents = contents.replace("{{ MIP_OPTIONS }}", mip_opt_line)
            contents = contents.replace("{{ DATA_FILE }}", data_file)
            f.write(contents)

        # 2b. building mip file
        target_bounds = gen_ampl_data_file_v2(df, preds, dep_scores, bin_probs, bin_pred_probs,
                                              bin_pred_pair_probs, target_counts, slack_percent,
                                              data_file, user_pred_ids=user_pred_ids)

        # 2c. running mip solver, checking for errors
        mip_answer_ids, mip_status, mip_output, mip_runtime = \
                get_ampl_result(model_file, mip_solver_timeout, print_details=(print_details or min_print_details))
        if mip_answer_ids:
            # getting answer query and resulting histogram
            answer_preds, meta['final_answer_pids'] = \
                get_preds_from_mip_solution(preds, mip_answer_ids, dep_scores=dep_scores, print_details=print_details)
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
            if print_details:
                print "query: {}".format(qs)
                print "len answer: {}".format(len(answer_df))
                print "Distrib of fld='{}' of full dataset: {}".format(target_fld, np.histogram(df[target_fld], adjusted_bin_edges)[0])
                print "Target bounds: {}".format(target_bounds)
                print "Distrib of fld='{}' w/ MIP solution's query applied: {}".format(target_fld, answer_distrib)

            # getting final meta data on solution
            meta['solution_found'] = True
            meta['solution_sp'] = sp
            meta['is_valid_solution'] = True
            for bin_id, (lower, upper) in enumerate(target_bounds):
                if answer_distrib[bin_id] < lower or answer_distrib[bin_id] > upper:
                    meta['is_valid_solution'] = False
                    break
            meta['exact_solution'] = (meta['final_answer_pids'] and tuple(meta['final_answer_pids']) == tuple(target_answer_pids))
            meta['semi_exact'] = meta['is_valid_solution'] and all(x in meta['final_answer_pids'] for x in target_answer_pids)
            if print_details:
                print "is_valid_solution = {}, exact? {}".format(yes_no(meta['is_valid_solution']), yes_no(meta['exact_solution']))

        meta['mip_statuses'].append(mip_status)
        meta['mip_runtimes'].append(mip_runtime)
        meta['mip_outputs'].append(mip_output)
        meta['slack_percents'].append(sp)
        meta['minos_opt_strs'].append(minos_opt_str)
        if meta['is_valid_solution']:
            break
        if mip_status == "timeout":
            print("WARNING: terminating on first timeout ({} sec)".format(mip_solver_timeout))
            break

    # 3. showing last output if no solution found
    if print_details and not meta['solution_found']:
        print("\n========== ERROR: no matches; last output: ==============")
        print mip_output

    # 4. returning meta
    return meta











###############################################################################
####################### GLPK-based Tiresias solver ############################

def gen_tiresias_data_file_v1(df, preds, target_counts, target_fld,
                              adjusted_bin_edges, slack_percent, output_file):
    """
    Generates data file for input to ampl
    """
    pred_ids = range(len(preds))
    bin_ids = range(len(target_counts))
    tuple_ids = df.index.tolist()

    # 1. mapping tuples to bins
    bin_tuple_ids = defaultdict(list)
    for bin_id, bin_start in enumerate(adjusted_bin_edges[0:-1]):
        bin_end = adjusted_bin_edges[bin_id + 1]
        bin_range = (bin_start, bin_end)
        tmp_df = df.query(Query.get_qs_pred(target_fld, 'range', bin_range))
        bin_tuple_ids[bin_id] = tmp_df.index.tolist()

    # 2. getting predicate mappings
    tuple_preds = defaultdict(list)
    for pred_id, pred in enumerate(preds):
        tmp_df = df.query(Query.get_pandas_query_from_preds([pred]))
        for tuple_id in tmp_df.index.tolist():
            tuple_preds[tuple_id].append(pred_id)
    # for pred_id, pred in enumerate(preds):
    #     tmp_df = df.query(Query.get_pandas_query_from_preds([pred]))
    #     tuple_preds[pred_id] = tmp_df.index.tolist()

    target_bounds = get_target_count_bounds(target_counts, slack_percent)
    with open(output_file, "w") as f:
        f.write(get_tiresias_baseline_data(tuple_ids, bin_ids, pred_ids, bin_tuple_ids,
                                           tuple_preds, target_counts, target_bounds))
    # 7. returning some meta data
    return target_bounds


def run_tiresias_solver_v1(df, preds, target_counts, target_fld, adjusted_bin_edges,
                           target_answer_pids, slack_percent=0.2, slack_increment=None, slack_max=1.0,
                           mip_solver_timeout=600, print_details=True, filename_uid=None,
                           syn_data_mode="v1"):
    """
    Generates mathprog/ampl file and gets results
    """
    meta = {}

    # 1. determining slack percents to use
    slack_percents = [slack_percent]
    if slack_increment:
        if not slack_increment or not slack_max:
            raise Exception("If using slack_max, must use slack_increment (and vice versa).")
        for sp in np.arange(slack_percent + slack_increment, slack_max, slack_increment):
            slack_percents.append(sp)
        if print_details:
            print "Slack percents: {}".format(slack_percents)

    # 2. running ampl for each slack percent, stopping once solution found
    meta['solution_found'] = False
    meta['is_valid_solution'] = False
    meta['semi_exact'] = False
    meta['exact_solution'] = False
    meta['solution_sp'] = None
    meta['final_answer_pids'] = None
    meta['mip_runtimes'] = []
    meta['mip_outputs'] = []
    meta['mip_statuses'] = []
    meta['slack_percents'] = []
    meta['minos_opt_strs'] = []
    for sp in slack_percents:
        if print_details:
            print "-------------------------------"
            print "Current slack = {}".format(sp)

        # 2a. determining model and data file to use
        if filename_uid is None:
            filename_uid = hash(dt.datetime.now())
        data_file = os.path.join(MIP_INPUT_FILE_DIR, syn_data_mode, "{}.slack{}.data".format(filename_uid, sp))
        model_file = os.path.join(MIP_INPUT_FILE_DIR, syn_data_mode, "{}.slack{}.mod".format(filename_uid, sp))
        cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "optimization-models"))
        with open(os.path.join(model_dir, "baseline-tiresias-con0.mod")) as f:
            model_file_contents = f.read()

        # 2b. writing model file (NOTE (FUTURE USE): change options here)
        with open(model_file, "w") as f:
            f.write(model_file_contents)

        # 2b. building mip file
        target_bounds = gen_tiresias_data_file_v1(df, preds, target_counts, target_fld,
                                                  adjusted_bin_edges, slack_percent, data_file)

        # 2c. running mip solver, checking for errors
        mip_answer_ids, mip_status, mip_output, mip_runtime = \
            get_glpk_result(model_file, data_file, mip_solver_timeout, print_details=print_details)
        if mip_answer_ids:
            # getting answer query and resulting histogram
            answer_preds, meta['final_answer_pids'] = \
                get_preds_from_mip_solution(preds, mip_answer_ids, print_details=print_details)
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
            if print_details:
                print "query: {}".format(qs)
                print "len answer: {}".format(len(answer_df))
                print "Distrib of fld='{}' of full dataset: {}".format(target_fld, np.histogram(df[target_fld], adjusted_bin_edges)[0])
                print "Target bounds: {}".format(target_bounds)
                print "Distrib of fld='{}' w/ MIP solution's query applied: {}".format(target_fld, answer_distrib)

            # getting final meta data on solution
            meta['solution_found'] = True
            meta['solution_sp'] = sp
            meta['is_valid_solution'] = True
            for bin_id, (lower, upper) in enumerate(target_bounds):
                if answer_distrib[bin_id] < lower or answer_distrib[bin_id] > upper:
                    meta['is_valid_solution'] = False
                    break
            meta['exact_solution'] = (meta['final_answer_pids'] and tuple(meta['final_answer_pids']) == tuple(target_answer_pids))
            meta['semi_exact'] = meta['is_valid_solution'] and all(x in meta['final_answer_pids'] for x in target_answer_pids)
            if print_details:
                print "is_valid_solution = {}, exact? {}".format(yes_no(meta['is_valid_solution']), yes_no(meta['exact_solution']))

        meta['mip_statuses'].append(mip_status)
        meta['mip_runtimes'].append(mip_runtime)
        meta['mip_outputs'].append(mip_output)
        meta['slack_percents'].append(sp)
        meta['minos_opt_strs'].append(None)
        if meta['is_valid_solution']:
            break

    # 3. showing last output if no solution found
    if print_details and not meta['solution_found']:
        print("\n========== ERROR: no matches; last output: ==============")
        print mip_output

    # 4. returning meta
    return meta












###############################################################################
#################### Emeril Hybrid (greedy + mip solver) ######################

def run_emeril_hybrid_v1(df, preds, dep_scores, bin_probs, bin_pred_probs,
                         bin_pred_pair_probs, target_counts, target_fld, adjusted_bin_edges,
                         target_answer_pids, slack_percent=0.2, max_preds=2,
                         slack_increment=None, slack_max=1.0,
                         mip_solver_timeout=600, print_details=False, filename_uid=None,
                         syn_data_mode="v1", minos_options=None, min_print_details=False,
                         solver='minos', user_pred_ids=None):
    """
    combines greedy with emeril
    """
    # trying greedy first
    meta = run_greedy_solver_v1(df, preds, target_counts, target_fld,
                                adjusted_bin_edges, target_answer_pids,
                                slack_percent=slack_percent,
                                print_details=print_details,
                                max_preds=max_preds)
    if meta['is_valid_solution']:
        if min_print_details:
            print "SOLVER USED: greedy"
        meta['hybrid_solver'] = 'greedy'
        return meta

    # otherwise, running emeril
    greedy_runtime = meta['mip_runtimes'][-1]
    meta = run_mip_solver_v2(df, preds, dep_scores, bin_probs, bin_pred_probs,
                             bin_pred_pair_probs, target_counts,
                             target_fld, adjusted_bin_edges, target_answer_pids,
                             slack_percent=slack_percent,
                             slack_increment=slack_increment,
                             slack_max=slack_max,
                             mip_solver_timeout=mip_solver_timeout,
                             print_details=print_details,
                             filename_uid=filename_uid,
                             syn_data_mode=syn_data_mode,
                             minos_options=minos_options,
                             min_print_details=min_print_details,
                             solver=solver,
                             user_pred_ids=user_pred_ids)
    meta['hybrid_solver'] = 'emerilIndep'
    if min_print_details:
        print "SOLVER USED: emerilIndep (runtime={:.3f} + {:.3f})".format(greedy_runtime, meta['mip_runtimes'][-1])
    for i in xrange(len(meta['mip_runtimes'])):
        meta['mip_runtimes'][i] += greedy_runtime
    return meta






###############################################################################
######################## Minos Option testing #################################

def minos_str_to_dict(minos_str):
    return {p: pv for p, pv in [x.split('=') for x in minos_str.split(' ')]}


def minos_dict_to_str(minos_options):
    return ' '.join(['{}={}'.format(p, minos_options[p]) for p in sorted(minos_options.keys())])


def test_minos_options(minos_str, orig_model_file, model_file):
    """
    gets ampl result for test.mod file
    """
    # build mod file
    with open(orig_model_file) as f:
        contents = f.read()
    contents = re.sub(r"option minos_options '.*'",
                      "option minos_options '{}'".format(minos_str),
                      contents)
    with open(model_file, 'w') as f:
        f.write(contents)

    # getting ampl result
    mip_timeout = 300
    try:
        mip_answer_ids, mip_status, mip_output, mip_runtime = \
                get_ampl_result(model_file, mip_timeout, print_details=False,
                                show_output_on_error=False)
    except SolverFailureSuperbasicsLimit:
        mip_answer_ids = None
        mip_status = 'superbasics_limit_failure'
    return mip_status, mip_answer_ids
