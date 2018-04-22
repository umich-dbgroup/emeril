/** SETS **/
set tuple_ids;
set bin_ids;
set pred_ids;
set bin_tuple_ids{bin_ids}, within tuple_ids;

/** PARAMS **/
param tuple_preds{tuple_ids, pred_ids};
param num_bins;
param num_preds;
param target_counts{bin_ids};
param target_bounds{bin_ids, {'lower', 'upper'}};

/** VARAIBLES **/
var preds {pred_ids} binary;
var not_preds {pred_ids} binary;
var tuples {tuple_ids} binary;
var tp_ors {tid in tuple_ids, pid in pred_ids} binary;
var tp_ands {tid in tuple_ids, pid in 0..(num_preds-1)} binary;
var bin_counts {bid in bin_ids}, >= target_bounds[bid,'lower'], <= target_bounds[bid,'upper'];
var num_chosen_preds integer, >= 1, <= 2;
var signal_dist;
var signal_dist_aux{bin_ids}; /* hack for abs() in objective */

/** CONSTRAINTS **/
/* defining not_preds as inverse of preds */
s.t. c_not_preds{pid in pred_ids}: preds[pid] = 1 - not_preds[pid];

/* creating tp_ors = (!p1 or tp1.1) statements */
s.t. c_tp_ors_a{tid in tuple_ids, pid in pred_ids}: not_preds[pid] <= tp_ors[tid, pid];
s.t. c_tp_ors_b{tid in tuple_ids, pid in pred_ids}: tuple_preds[tid, pid] <= tp_ors[tid, pid];
s.t. c_tp_ors_c{tid in tuple_ids, pid in pred_ids}: not_preds[pid] + tuple_preds[tid, pid] >= tp_ors[tid, pid];

/* base for combining tp_ors into (t0 = tp_ors[0] and ...) statements */
c_tp_ands1{tid in tuple_ids}: tp_ors[tid, 0] >= tp_ands[tid, 0];
c_tp_ands2{tid in tuple_ids}: tp_ors[tid, 0] + tp_ors[tid, 1] <= tp_ands[tid, 0] + 1;

/* chaining ands to combine tp_ors into (t1 = tp_ors[1] and ...) statements */
c_tp_ands3{tid in tuple_ids, xid in 0..(num_preds-3)}: tp_ands[tid, xid] >= tp_ands[tid, xid + 1];
c_tp_ands4{tid in tuple_ids, xid in 0..(num_preds-2)}: tp_ors[tid, xid + 1] >= tp_ands[tid, xid];
c_tp_ands5{tid in tuple_ids, xid in 0..(num_preds-3)}: tp_ands[tid, xid] + tp_ors[tid, xid + 2] <= tp_ands[tid, xid + 1] + 1;
c_tuples{tid in tuple_ids}: tuples[tid] = tp_ands[tid, num_preds - 2];

/* bin counts and num preds statements */
s.t. c3{bid in bin_ids}: bin_counts[bid] = sum{tid in bin_tuple_ids[bid]} tuples[tid];
s.t. c4: num_chosen_preds = sum{pid in pred_ids} preds[pid];

/* defines signal distance as sum of differences vs targets */
s.t. c5{bid in bin_ids}: signal_dist_aux[bid] >= bin_counts[bid] - target_counts[bid];
s.t. c6{bid in bin_ids}: signal_dist_aux[bid] >= target_counts[bid] - bin_counts[bid];
s.t. c7: signal_dist = sum{bid in bin_ids} signal_dist_aux[bid] / num_bins;

/** OBJECTIVE **/
/*minimize z: num_chosen_preds + signal_dist;*/
minimize z: signal_dist;

/** SOLVE **/
solve;
printf "\n";
printf "******************************\n";
printf "Chosen pred_ids (%i total): \n", num_chosen_preds;
printf {pid in pred_ids} (if preds[pid] > 0.0 then "%i, " else ""), pid;
printf "\n----------------------------\n";
printf "Bin target => actual counts:\n";
printf {bid in bin_ids} "bin %i: %i => %.2f\n", bid, target_counts[bid], bin_counts[bid];
printf "----------------------------\n";
printf "signal_dist: %.2f\n", signal_dist;
/*
printf "------------------------------\n";
printf "Tuples:\n";
printf {tid in tuple_ids} (if tuples[tid] > 0.0 then "%i = %.4f\n" else ""), tid, tuples[tid];
printf "------------------------------\n";
printf "Preds:\n";
printf {pid in pred_ids} (if preds[pid] > 0.0 then "%i = %.4f\n" else ""), pid, preds[pid];
printf "------------------------------\n";
printf "TP Ors:\n";
printf {tid in tuple_ids, pid in pred_ids} "%i, %i = %.4f\n", tid, pid, tp_ors[tid, pid];
printf "------------------------------\n";
printf "TP Ands:\n";
printf {tid in tuple_ids, pid in 1..(num_preds-1)} "%i, %i = %.4f\n", tid, pid, tp_ands[tid, pid];
*/
printf "******************************\n";

end;
