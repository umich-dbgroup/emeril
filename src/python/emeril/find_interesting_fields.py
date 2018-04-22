from . import *

###############################################################################
################## Find relevant distrib triggering queries ###################

def get_changed_fields(df, fields_to_filter, fields_to_search,
                       min_corr=DISTRIB_CHANGE_MIN_CORR):
    start = timer()
    all_changed = []
    print("Starting search for changed fields ({} fields to filter with)..."
          .format(len(fields_to_filter)))
    for fld_id, fld in enumerate(fields_to_filter):
        # to start, only look at rows with values for current fld
        cur_df = df[~np.isnan(df[fld])]
        for mode in ('lt', 'gte'):
            if mode == 'lt':
                keys = cur_df[fld] < cur_df[fld].median()
            elif mode == 'gte':
                keys = cur_df[fld] >= cur_df[fld].median()
            filtered_cur_df = cur_df[keys]
            if len(cur_df) == 0:
                continue
            changed = find_distrib_changes(df1=cur_df,
                                           df2=filtered_cur_df,
                                           fields_to_search=fields_to_search,
                                           min_corr=min_corr)
            if changed:
                for (changed_fld, corr, count_ratio, rmse, len_df1, len_df2) in changed:
                    all_changed.append((fld, changed_fld, mode, corr, count_ratio,
                                        rmse, len_df1, len_df2))
        if fld_id % 50 == 0:
            print("At fld_id={}, total changed={} (at {:.2f} secs)".format(fld_id, len(all_changed), timer() - start))
    print("Done at {:.2f} secs; total changed candidates: {}".format(timer() - start, len(all_changed)))
    return all_changed


def describe_changed_field(df, data_item, plot_signals=True, fld_dict=None):
    fld, changed_fld, mode, corr, count_ratio, rmse, len_df1, len_df2 = data_item

    cur_df = df[~np.isnan(df[fld])]
    if mode == 'lt':
        filtered_cur_df = cur_df[cur_df[fld] < cur_df[fld].median()]
    else:
        filtered_cur_df = cur_df[cur_df[fld] >= cur_df[fld].median()]

    df1 = cur_df[~np.isnan(cur_df[changed_fld])]
    df2 = filtered_cur_df[~np.isnan(filtered_cur_df[changed_fld])]

    sig1x, sig1y = get_distrib_signal(df1[changed_fld], data_min=df1[changed_fld].min(), data_max=df1[changed_fld].max(), ret_x=True)
    sig2x, sig2y = sig1x, get_distrib_signal(df2[changed_fld], grid=sig1x)

    print("fld={}, changed_fld={}, mode={}, corr={:.2f}, "
          "count_ratio={:.2f}, rmse={:.2f}, len_df1={}, len_df2={}, len_set2_uniques={}"
          .format(fld, changed_fld, mode, corr, count_ratio, rmse,
                  len_df1, len_df2, len(set(df2[changed_fld]))))
    if fld_dict:
        print("fld = {}".format(fld_dict[fld]))
        print("Changed fld = {}".format(fld_dict[changed_fld]))


    if len(set(df1[changed_fld])) <= 2 or len(set(df2[changed_fld])) <= 2:
        print("- WARNING.. len(set(sig1)))={}, set(sig2)={}"
              .format(len(set(df1[changed_fld])), set(df2[changed_fld])))

    if plot_signals:
        plt.plot(sig1x, sig1y, label='pre-filter')
        plt.plot(sig2x, sig2y, label='post-filter')
        plt.legend()
        plt.show()


def backup_changed(changed, name):
    """
    used to backup list of changed fields (results from get_changed_fields)
    """
    with open(os.path.join(CHANGED_PICKLE_PATH, name), 'wb') as f:
        pickle.dump(changed, f, -1)


def restore_changed(name):
    """
    used to restore list of changed fields (results from get_changed_fields)
    """
    with open(os.path.join(CHANGED_PICKLE_PATH, name), 'rb') as f:
        return pickle.load(f)
