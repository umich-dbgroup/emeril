# Emeril #

This project contains source code for the publication, *Constraint-based Explanation and Repair of Filtered-based Transformations*, a paper to be presented at VLDB'18. **Caveat:** I haven't had a chance to test this version of the code (cleaned up to make it easier for others to run), so if you have any issues, ping me and I'll try to help.



## Using Emeril for Real-World Datasets ##

- **Caveat:** Code is only set for running experiments, so a bit of hacking needed before any dataset can be used.
- Process:
  1. Install python packages listed in requirements.txt
  2. Modify `get_real_world_data` and `add_real_world_syn_answer_v1` in src/emeril/realworld.py to define/load your data in a similar manner as the other datasets.
  3. Run following (or look at `test_real_world_v1` in src/emeril/realworld.py):
    ```bash
    $ cd src/python
    $ python
    > meta = test_real_world_v1(
        dataset_name="[your dataset identifier]",
        pred_dep_percent=0.01,
        print_details=True,
        mip_solver_timeout=3600,
        system="emerilTriHybrid",
        pred_mode='naive',
        max_preds=2,
        rw_data_mode="rw1"
    )
    ```



## Running Synthetic Data Experiments ##

### Baseline Experiments ###
```bash
## n-choose-k (max_preds=2) ##
~/emeril/src/python/experiments/syndata_runs_v2.py --mode=corr \
  --pred_dep_percents="0.0" \
  --num_runs=10 --slack_percent=0.2 --random_data \
  --system=n_choose_k --max_preds=2

## tiresias ##
~/emeril/src/python/experiments/syndata_runs_v2.py --mode=corr \
  --pred_dep_percents="0.0" \
  --num_runs=10 --slack_percent=0.2 --random_data \
  --system=tiresias --mip_solver_timeout=3600
```


### Running Emeril Experiments ###
1. Pre-loading pred-pair data (optional, but saves time on first run w/ data)
```bash
~/emeril/src/python/experiments/gen_exp_data.py \
  --mode="rows" --num_runs="1" --syn_data_mode="v9" --n_processes="32" \
  --default_corr="0.2" --preds="1000"
```

2. Get synthetic pred results
```bash
# if using ampl, set path to where ampl binary located
$ export EMERIL_AMPL_DIR="path/to/ampl/binary/directory"

# if using glpk (i.e., for baseline test), set path to where glpk binary located
$ export EMERIL_GLPK_DIR="path/to/glpk/binary/directory"

# synthetic runs
~/emeril/src/python/experiments/syndata_runs_v2.py --mode="preds" \
  --pred_dep_percents="0.1" \
  --preds="1000" \
  --use_meta_cache \
  --num_runs="1" --slack_percent="0.2" --random_data --max_preds="2" \
  --system="emerilTriHybrid" --minos_options="dynamic" \
  --mip_solver_timeout="3600" --syn_data_mode="v9"
```
