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
from . import *

###############################################################################
########################### Initialization ####################################

class DatabaseMeta:
    """
    Collects and stores meta data about database tables
    """
    tables = None
    numeric_fields = None
    numeric_int_fields = None
    numeric_float_fields = None
    categorical_fields = None
    field_importance_scores = None
    field_types = None
    field_bins = None
    io_ratios = None
    field_sub_bins = None

    def __init__(self):
        self.tables = []
        self.numeric_fields = defaultdict(list)
        self.numeric_int_fields = defaultdict(list)
        self.numeric_float_fields = defaultdict(list)
        self.categorical_fields = defaultdict(list)
        self.field_importance_scores = defaultdict(list)
        self.field_types = {}
        self.field_bins = {}
        self.io_ratios = defaultdict(dict)
        self.field_sub_bins = defaultdict(dict)

    @staticmethod
    def has_backup(name):
        """
        checks if backup available
        """
        if not name:
            raise Exception("Must specify a name.")
        return os.path.exists(os.path.join(DB_META_PICKLE_PATH, name))

    def backup(self, name):
        """
        used to backup list of changed fields (results from get_changed_fields)
        """
        if not name:
            raise Exception("Must specify a name.")
        with open(os.path.join(DB_META_PICKLE_PATH, name), 'wb') as f:
            data = (
                self.tables,
                self.numeric_fields,
                self.numeric_int_fields,
                self.numeric_float_fields,
                self.categorical_fields,
                self.field_importance_scores,
                self.field_types,
                self.field_bins,
                self.io_ratios,
                self.field_sub_bins,
            )
            pickle.dump(data, f, -1)

    @staticmethod
    def restore(name):
        """
        used to restore list of changed fields (results from get_changed_fields)
        """
        if not name:
            raise Exception("Must specify a name.")
        with open(os.path.join(DB_META_PICKLE_PATH, name), 'rb') as f:
            d = DatabaseMeta()
            (
                d.tables,
                d.numeric_fields,
                d.numeric_int_fields,
                d.numeric_float_fields,
                d.categorical_fields,
                d.field_importance_scores,
                d.field_types,
                d.field_bins,
                d.io_ratios,
                d.field_sub_bins,
            ) = pickle.load(f)
            return d

    def analyze_dfs(self, dfs, ignore_fields=None):
        """Takes dictionary of {table_name : df, ...} to be analyzed."""
        start = timer()
        print("Analying {} dataframes...".format(len(dfs)))

        if self.numeric_fields or self.categorical_fields:
            raise Exception("Already processed since fields defined.")

        for table_name, df in dfs.iteritems():
            print("At table {} ({:.3f} seconds)...".format(table_name, timer() - start))
            self.tables.append(table_name)
            for i, fld in enumerate(df.columns):
                if ignore_fields and fld in ignore_fields:
                    continue

                fld_type = df[fld].dtype
                if fld_type in ('int64', 'float64'):
                    unique_vals = set(df[~np.isnan(df[fld])][fld])
                    num_unique = len(unique_vals)
                    # unique_ratio = num_unique / len(df)

                    if num_unique > MAX_CATEGORICAL_VALUES and fld_type in ('int64', 'float64'):
                        self.numeric_fields[table_name].append(fld)
                        if fld_type == 'int64':
                            self.numeric_int_fields[table_name].append(fld)
                        else:
                            self.numeric_float_fields[table_name].append(fld)
                        self.field_types[fld] = 'numeric'

                        # adding bins for numeric field
                        _, self.field_bins[fld] = np.histogram(df[~np.isnan(df[fld])][fld])  # all but last half-open
                    elif num_unique > MAX_CATEGORICAL_VALUES:
                        pass  # ignoring field
                    else:
                        self.categorical_fields[table_name].append(fld)
                        self.field_types[fld] = 'categorical'
                        self.field_bins[fld] = sorted(list(unique_vals))

                        # adding one last bin to account for edges needing +1
                        # (only adding if there are bins)
                        if self.field_bins[fld]:
                            self.field_bins[fld].append(self.field_bins[fld][-1] + 1)
                elif fld_type == 'O':
                    pass  # TODO: add support for char/str categorical (convert to numeric first)
                else:
                    raise Exception("Unsupported field type: {} ({}).".format(fld_type, fld))
        print("Done analying dataframes ({:.3f} seconds)...".format(timer() - start))

    def print_summary(self):
        for tbl in self.tables:
            print("{} summary: num int fields={}, float fields={}, categorical fields={}"
                  .format(tbl, len(self.numeric_int_fields[tbl]),
                          len(self.numeric_float_fields[tbl]),
                          len(self.categorical_fields[tbl])))

    def load_importance_from_google_scholar(self, project_keyword, double_under_hack=True):
        """
        Fills in field_importance_scores with
        (fld, <count fld on first page>, <total num search results>) tuples.
        """
        if os.path.exists(SCRAPER_CACHE_FILE):
            with open(SCRAPER_CACHE_FILE) as f:
                scraper_cache = pickle.load(f)
        else:
            scraper_cache = {}

        self.field_importance_scores = defaultdict(list)  # resetting
        cur_pos = 0
        for table in self.tables:
            print("=============== Grabbing fields for {} =================".format(table))
            cur_fields = self.numeric_fields[table] + self.categorical_fields[table]
            for fld in cur_fields:
                cur_pos += 1

                if double_under_hack:
                    website_fld = fld.replace('__', '.')
                else:
                    website_fld = fld

                url = 'http://scholar.google.com/scholar?hl=en&q={}+{}'\
                      .format(project_keyword, website_fld)

                if url in scraper_cache:
                    html = scraper_cache[url]
                    # print("{}. Grabbing cache of {}".format(cur_pos, url))
                else:
                    sleep(random.uniform(0.1, 0.5))
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
                    }
                    cookies = {

                    }
                    print("{}. Loading {}".format(cur_pos, url))
                    r = requests.get(url, headers=headers, cookies=cookies)
                    if r.status_code != 200:
                        raise Exception("Url={} has status code = {} (try setting cookies?)".format(url, r.status_code))
                    elif "Sorry, we can't verify that you're not a robot when JavaScript is turned off" in r.text:
                        raise Exception("Url={} has google js-disabled error.".format(url))
                    else:
                        html = r.text
                        scraper_cache[url] = html
                        with open(SCRAPER_CACHE_FILE, 'w') as f:
                            pickle.dump(scraper_cache, f, -1)

                # get num results & occurrences of field in HTML
                m = re.search('(About )?([\d,]+) results?', html)
                if m:
                    # num results
                    num_results = int(m.group(2).replace(',', ''))

                    # num hits of keyword
                    # num_field_hits = html.count(website_fld)
                    num_field_hits = strip_tags(html).count(website_fld)
                elif re.search('Your search .* did not match any articles.', html):
                    num_field_hits = 1  # actually has 2, but don't want to rank too high
                    num_results = 0
                else:
                    raise Exception("Something wrong with url={}\n\nRaw text:\n\n\n{}".format(url, html))

                # checking if page contains 'Did you mean'
                did_you_mean_penalty = 0
                if re.search('Did you mean:', html):
                    did_you_mean_penalty = -1

                scores_tuple = (fld, (num_field_hits, did_you_mean_penalty, num_results))
                self.field_importance_scores[table].append(scores_tuple)

    def get_fields_by_importance(self):
        """
        returns list of (table, fld, score) tuples, ordered by best score
        """
        scored_fields = []
        for table in self.field_importance_scores.keys():
            for fld, score in self.field_importance_scores[table]:
                scored_fields.append((table, fld, score))
        scored_fields.sort(key=lambda x: x[2], reverse=True)
        return scored_fields

    def get_searchable_fields(self):
        searchable = []
        for table, fields in self.numeric_fields.iteritems():
            searchable += fields
        for table, fields in self.categorical_fields.iteritems():
            searchable += fields
        return searchable

    def get_field_type(self, fld):
        return self.field_types[fld]

    def get_field_bins(self, fld):
        return self.field_bins[fld]

    def preload_field_in_out_ratios(self, df, fld, use_sb_edges=False):
        """
        REMINDER: pandas negation filter includes NaN's; filter excludes NaN's
        """
        fld_bin_edges = self.get_field_bins(fld)
        fld_type = self.get_field_type(fld)
        searchable_fields = self.get_searchable_fields()  #['RIDRETH1', 'RIAGENDR', 'RIDAGEYR', 'BPXSY1', 'LBXSCH']

        if use_sb_edges and fld not in self.field_sub_bins:
            raise Exception("sub_bin meta must be preloaded for use_sb_edges")

        start = timer()
        for bin_id in range(len(fld_bin_edges) - 1):
            print("- at bin_id={}, {:.2f} seconds elapsed".format(bin_id, timer() - start))

            # 1. loading bins' in/out lengths and query strings
            if fld_type == 'numeric':
                bin_qs = '({} >= {} and {} < {})'.format(fld, fld_bin_edges[bin_id], fld, fld_bin_edges[bin_id + 1])
            elif fld_type == 'categorical':
                bin_qs = '({} == {})'.format(fld, fld_bin_edges[bin_id])
            orig_in_len = float(len(df.query(bin_qs)))
            orig_out_len = float(len(df) - orig_in_len)
            if orig_in_len == 0:
                print("bin={} has 0 in-bin records. qs={}".format(bin_id, bin_qs))
                continue

            # 2. go through all subfields' bins
            self.io_ratios[fld][bin_id] = []
            for sf in searchable_fields:
                if fld == sf:
                    continue

                # determining sub-bin edges to use (sub-bin vs. all data)
                sf_type = self.get_field_type(sf)
                if use_sb_edges:
                    if sf not in self.field_sub_bins[fld][bin_id]:
                        continue
                    (_, sb_edges, _, _) = self.field_sub_bins[fld][bin_id][sf]
                else:
                    sb_edges = self.get_field_bins(sf)

                # next, go through all sub-bins, noting i/o ratios
                for sb_id in range(len(sb_edges) - 1):
                    # filtering full df on sub-bin predicate
                    if sf_type == 'numeric':
                        sf_qs = '~({} >= {} and {} < {})'.format(sf, sb_edges[sb_id], sf, sb_edges[sb_id + 1])
                        sf_pred = (sf, 'range', (sb_edges[sb_id], sb_edges[sb_id + 1]))
                    elif sf_type == 'categorical':
                        sf_qs = '~({} == {})'.format(sf, sb_edges[sb_id])
                        sf_pred = (sf, '==', sb_edges[sb_id])
                    else:
                        raise Exception("unsupported field type")
                    post_rm_df = df.query(sf_qs)

                    # skipping sub-bins that remove nothing or everything
                    if len(post_rm_df) == 0 or len(post_rm_df) == len(df):
                        continue

                    in_kept = len(post_rm_df.query(bin_qs))
                    out_kept = len(post_rm_df) - in_kept
                    in_kept_p = in_kept / orig_in_len
                    out_kept_p = out_kept / orig_out_len
                    in_rm = orig_in_len - in_kept
                    out_rm = orig_out_len - out_kept
                    in_rm_p = in_rm / orig_in_len
                    out_rm_p = out_rm / orig_out_len

                    # skipping if didn't remove any in-bin rows
                    if in_rm == 0:
                        continue

                    io_ratio = get_fscore(in_rm_p, out_kept_p, beta=4.0)
                    self.io_ratios[fld][bin_id].append((io_ratio, in_rm_p, out_kept_p, sf_pred))

            # 3. sort ratios
            self.io_ratios[fld][bin_id].sort(reverse=True, key=lambda x: x[0])
            print("\nBIN={} IO RATIOS: {}".format(bin_id, bin_qs))
            for io_ratio, in_rm_p, out_kept_p, pred in self.io_ratios[fld][bin_id][0:5]:
                tmp_fld, tmp_fld_op, tmp_value = pred
                qstr = Query.get_qs_pred(tmp_fld, tmp_fld_op, tmp_value, negation=True)
                print("io_ratio={:.2f}, in_rm_p={:.2f}, out_kept_p={:.2f}, pred={}"
                      .format(io_ratio, in_rm_p, out_kept_p, qstr))

    def get_field_in_out_ratios(self, fld):
        return self.io_ratios[fld]

    def preload_field_sub_bins(self, df, fld):
        fld_bin_edges = self.get_field_bins(fld)
        fld_type = self.get_field_type(fld)
        searchable_fields = self.get_searchable_fields() #['RIAGENDR', 'RIDAGEYR', 'BPXSY1', 'LBXSCH']
        start = timer()
        print("Pre-loading sub-bin meta for field={}...".format(fld))
        for bin_id in range(len(fld_bin_edges) - 1):
            print("- at bin_id={}, {:.2f} seconds elapsed".format(bin_id, timer() - start))

            # 1. loading current bin's data
            if fld_type == 'numeric':
                bin_qs = '({} >= {} and {} < {})'.format(fld, fld_bin_edges[bin_id], fld, fld_bin_edges[bin_id + 1])
            elif fld_type == 'categorical':
                bin_qs = '({} == {})'.format(fld, fld_bin_edges[bin_id])
            else:
                raise Exception("unsupported field type")
            bin_df = df.query(bin_qs)
            bin_len = float(len(bin_df))

            # 2. initializing bin list
            self.field_sub_bins[fld][bin_id] = {}

            # 3. determining sub_bins for each attribute
            for sf in searchable_fields:
                if fld == sf:
                    continue

                # 3a. get current attrib data
                sb_data = bin_df[map(lambda x: np.isreal(x) and x >= 0.0, bin_df[fld])][fld]
                sb_len = float(len(sb_data))
                if sb_len == 0:
                    continue

                # 3b. get sub-histogram for current attrib on current bin
                sf_type = self.get_field_type(sf)
                if sf_type == 'numeric':
                    sb_counts, sb_edges = np.histogram(sb_data)
                elif fld_type == 'categorical':
                    # adding one last bin to account for edges needing +1
                    # (only adding if there are bins)
                    sb_edges = sorted(list(set(sb_data)))
                    if sb_edges:
                        sb_edges.append(sb_edges[-1] + 1)
                    sb_counts, sb_edges = np.histogram(sb_data, bins=sb_edges)
                else:
                    raise Exception("unsupported field type")

                # 3c. adding sub-bin meta data to list
                sb_percents = [x / sb_len for x in sb_counts]
                b_percents = [x / bin_len for x in sb_counts]
                sb_meta = (sb_counts, sb_edges, sb_percents, b_percents)
                self.field_sub_bins[fld][bin_id][sf] = sb_meta

    def get_field_sub_bins(self, fld):
        return self.field_sub_bins[fld]
