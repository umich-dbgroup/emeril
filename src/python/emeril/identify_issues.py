from . import *

###############################################################################
########################### Candidate Gen #####################################

def find_distrib_changes(df1, df2, fields_to_search,
                         min_corr=DISTRIB_CHANGE_MIN_CORR):
    main_df = df1 if len(df2) <= len(df1) else df2
    reduced_df = df2 if len(df2) <= len(df1) else df1
    df1 = main_df
    df2 = reduced_df
    changed = []
    num_empty_df_err = 0
    num_sig_gen_err = 0
    for fld in fields_to_search:
        try:
            corr, count_ratio, rmse, len_df1, len_df2, kl_div = compare_distribs(df1, df2, fld, include_kl_div=True)
            if corr < min_corr:
                changed.append((fld, corr, count_ratio, rmse, len_df1, len_df2))
        except EmptyDataFrameError:
            num_empty_df_err += 1
            pass
        except SignalGenerationError:
            num_sig_gen_err += 1
            pass
        except Exception as e:
            print("Error with fld={}, error=".format(fld, e))
            raise
    print("num_empty_df_err={}, num_sig_gen_err={}".format(num_empty_df_err, num_sig_gen_err))
    return changed


def identify_query_distrib_changes(df, query, fields_to_search,
                                   min_corr=DISTRIB_CHANGE_MIN_CORR):
    """
    finds distrib changes from applying query
    """
    # 1. check for distrib changes
    filter_query = query.get_pandas_query()
    print("Filter query = {}, len(fields_to_search)={}".format(filter_query, len(fields_to_search)))
    filtered_df = df.query(filter_query)
    return find_distrib_changes(df, filtered_df, fields_to_search, min_corr=min_corr)

    ## NOTE: already implemented in find_interesting_fields.py



def sort_query_issues(db_meta, issues):
    """
    sorts issues based on field-importance (and other params?)
    """
