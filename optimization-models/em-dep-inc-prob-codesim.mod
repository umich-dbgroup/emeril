/**
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
**/

/** OPTIONS **/
option solver '{{ MIP_SOLVER }}';
{{ MIP_OPTIONS }}
/*
option presolve 0;
option reset_initial_guesses 1;
option dual_initial_guess 0;
*/

/** SETS **/
set bin_ids;
set pred_ids;
set constraint_ids;
set pred_constraints{constraint_ids} within pred_ids;
set user_pred_ids;

/** PARAMS **/
param num_bins;
param total_rows;
param bin_probs{bin_ids};
param bin_pred_probs{pred_ids, bin_ids};
param bin_pred_counts{pred_ids, bin_ids};
param target_counts{bin_ids};
param target_bounds{bin_ids, {'lower', 'upper'}};
param first_combo_pid;
param user_pred_sims{pred_ids, user_pred_ids};

/** VARS **/
var preds{pred_ids}, binary;
var bin_counts {bid in bin_ids} >= target_bounds[bid,'lower'], <= target_bounds[bid,'upper'];
var num_chosen_preds >= 1.0, <= 2.0;
var signal_dist;
var user_pred_dists{upid in user_pred_ids};
var user_pred_dist;

/** CONSTRAINTS **/
s.t. c1{bid in bin_ids}: bin_counts[bid] = total_rows * bin_probs[bid] * exp(sum{pid in pred_ids} (preds[pid] * log(bin_pred_probs[pid, bid] + 0.0000000001)));
s.t. c2: num_chosen_preds = sum{pid in pred_ids} preds[pid];
s.t. c3{cid in constraint_ids}: sum{pid in pred_constraints[cid]} preds[pid] <= 1;
s.t. c4: signal_dist = sum{bid in bin_ids} abs(target_counts[bid] - bin_counts[bid]);

/* defines code distance using code-sim matrix w/ user's original predicates, plus num preds */
s.t. c5{upid in user_pred_ids}: user_pred_dists[upid] = sum{pid in pred_ids} (preds[pid] * (1 - user_pred_sims[pid, upid]));
s.t. c6: user_pred_dist = sum{upid in user_pred_ids} user_pred_dists[upid];

/** OBJECTIVE **/
minimize z: signal_dist + user_pred_dist;

/** SOLVE **/
data "{{ DATA_FILE }}";
solve;
printf "\n";
printf "******************************\n";
printf "Note: first combo pid: %i\n", first_combo_pid;
printf "Chosen pred_ids (%i total): \n", num_chosen_preds;
printf {pid in pred_ids} (if preds[pid] >= 0.5 then "%i, " else ""), pid;
printf "\n----------------------------\n";
printf "Bin target => actual counts:\n";
printf {bid in bin_ids} "bin %i: %i => %.2f\n", bid, target_counts[bid], bin_counts[bid];
printf "----------------------------\n";
printf "signal_dist: %.2f\n", signal_dist;
printf "user_pred_dist: %.2f\n", user_pred_dist;
printf "----------------------------\n";
printf {pid in pred_ids} (if preds[pid] >= 0.5 then "%.4f: %i\n" else ""), preds[pid], pid;
printf "******************************\n";

end;
