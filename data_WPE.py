import os
import logging
# Configure logging to show messages with INFO level or higher
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
import traceback
import pickle
import numpy as np
import yaml
from torch.utils.data import Dataset
from typing import List
import pandas as pd
from pandas import DataFrame


class EHRAuditLogDataSet(Dataset):
    """
    Dataset for Epic EHR audit log data.

    Assumes that the data is associated with a unique WPE committed by a single physician.
    Each sequence consists of EHR actions taken within the preceding hour before an ordering event (ie, case or control).
    Time deltas are calculated with respect to the preceding event.
    """

    def __init__(
            self,
            yaml_config_path: str,
            root_dir: str,
            session_sep_min: int = 5,
            pat_col: str = "PAT_ID",
            timestamp_col: str = "ACCESS_TIME",
            timestamp_sort_cols: List[str] = ["ACCESS_TIME", "ACCESS_INSTANT"],
            event_type_cols: List[str] = ["METRIC_NAME"],
            prov_col: str = "USER_ID",
            log_name: str = None,
            # timestamp_spaces: List[float] = None,
            cache: str = None,
            reset_cache: bool = False,
            clean_auditLogs: bool = False,
            unit_string: str = "session",
            tokenized_dataset=None,
            len_tokens: int = None,
            num_fields: int = 4,
            wpe_id: int = None,
            action_token_map: dict = None
    ):
        with open(yaml_config_path) as f:
            self.config = yaml.safe_load(f)
        # self.seqs = []
        self.len = None
        self.provider = os.path.basename(root_dir)
        self.session_sep_min = session_sep_min
        self.pat_col = pat_col
        self.timestamp_col = timestamp_col
        self.event_type_cols = event_type_cols
        self.log_name = log_name
        self.root_dir = root_dir
        # self.timestamp_spaces = timestamp_spaces
        self.timestamp_sort_cols = timestamp_sort_cols
        self.prov_col = prov_col
        # self.rowStrings = []
        self.sessionStrings = []
        self.clean_auditLogs = clean_auditLogs
        self.reset_cache = reset_cache
        self.unit_string = unit_string
        self.tokenized_dataset = tokenized_dataset
        self.len_tokens = len_tokens
        self.num_fields = num_fields
        self.wpe_id = str(wpe_id)
        self.action_token_map = action_token_map

        if cache is not None:
            self.cache = cache
        else:
            self.cache = None

    def load(self):
        """
        Load the dataset from either a log file or a cache.
        """
        # cache_path = os.path.normpath(os.path.join(self.root_dir, self.cache))
        # if not self.reset_cache and self.cache and os.path.exists(cache_path):
        #     # print("Loading cached dataset from {}".format(cache_path))
        #     self.load_from_cache()
        # else:
        #     # print("Loading & Prepping dataset from scratch.")
        #     self.load_from_log()

        # [NOTE] Disabling cache-based loading for now. Current pipeline doesn't save sessionStrings.pkl at initial loading
        # print("Loading & Prepping dataset from scratch.")
        self.load_from_log()


    def load_from_log(self):
        """
        Load the dataset from a log file.
        """
        # print(f'Loading {self.provider} from {self.log_name}...\n')
        path = os.path.normpath(os.path.join(self.root_dir, self.log_name))
        df = pd.read_parquet(path)

        # Sanity check: auto-skip files with missing columns
        expected_cols = set(['METRIC_ID']+self.event_type_cols + [self.pat_col] + self.timestamp_sort_cols + [self.prov_col])
        missing_cols = expected_cols - set(df.columns)
        if missing_cols or len(df)==0:
            logging.warning(f"Skipping {self.log_name} due to missing columns: {missing_cols}")
            self.df = pd.DataFrame()  # Empty DataFrame fallback
            self.rowStrings, self.sessionStrings = [], []
            return

        # Sanity Check: Ensure that timestamp_col is in timestamp_sort_cols
        if self.timestamp_col not in self.timestamp_sort_cols:
            raise ValueError(f"timestamp_col {self.timestamp_col} must be in timestamp_sort_cols")

        # Keep only the necessary columns
        # Currently, keeping 5: METRIC_NAME, PAT_ID, ACCESS_TIME, ACCESS_INSTANT, USER_ID
        # use this if conditional logic when loading in audit log datasets for the control sample group
        if ('control' in self.log_name) & ('control_idx' in df.columns):
            # print(f"[DEBUG] {self.log_name}: unique control_idx values = {df['control_idx'].unique()}")
            df = df[['METRIC_ID']+self.event_type_cols + [self.pat_col] + self.timestamp_sort_cols + [self.prov_col] + ['control_idx']]
            # keep N samples from all controls for a desired 1:N matching
            sample_n=self.config.get("sample_n_controls")
            if self.config.get("sample_n_controls") is not None:
                # For test WPEs with fewer than N matched controls, all available controls are retained.
                # which would make it 1:M matched (M<N).
                if df['control_idx'].nunique() > sample_n:
                    ## Option 1: randomly sample controls
                    # df = df.sample(n=sample_n, random_state=self.config.get("random_seed"))
                    ## Option 2: take the most recent n controls (from WPE case event time)
                    df = df[df['control_idx'].isin(sorted(df['control_idx'].unique())[-sample_n:])]
                    # MUST reset index for correct session slicing later.
                    df = df.reset_index(drop=True)
                # else:
                #     logging.info(
                #         f"[WPE {self.log_name}] has only {df['control_idx'].nunique()} controls < desired N={sample_n}")
                    # Later, count these log lines in the terminal output / logged txt file to count how many WPEs had less than N matchable controls.
                    # $ grep "has only" your_output_log_file.txt | wc -l
        else:
            df = df[['METRIC_ID']+self.event_type_cols + [self.pat_col] + self.timestamp_sort_cols + [self.prov_col]]

        # Convert timestamp column to a consistent numeric or datetime format
        if df[self.timestamp_col].dtype == np.dtype("O"):  # string format
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df[self.timestamp_col] = df[self.timestamp_col].astype(np.int64) // 10 ** 9  # seconds since epoch


        if self.unit_string == "all": # use the entire block sequence preceding an order
            # Timedelta_style1: diff(current, previous)
            time_deltas = df.loc[:, self.timestamp_col].diff(periods=1)
            # Timedelta_style2: diff(current, next)
            # time_deltas = df.loc[:, 'ACCESS_INSTANT'].diff(periods=-1)*-1

            # Must impute first row's timedelta with an artificial value > SESSION_INTERVAL to denote session start
            time_deltas.fillna(np.nan, inplace=True)

            df['TIME_DELTA'] = time_deltas.dt.total_seconds()
            

            def bucket_time_delta(seconds, threshold=60):
                # 60 sec is reasonable cutoff.
                # Beyond that may cause multi-token fragmentation
                # Fine-binning for 0-60 seconds
                if seconds <= threshold:
                    return int(seconds)  # fine-grained for rapid actions
                # Log-binning for >60 seconds
                else:
                    return int(np.log1p(seconds)) + threshold  # coarser for large delays; log1p = log(seconds + 1) safe for 0

            ## if using bucket_time_delta (more applicable for word-based free-text tokenization approach)
            # df['TIME_DELTA'] = df['TIME_DELTA'].apply(lambda x: '<FIRST_ROW>' if pd.isnull(x) else str(bucket_time_delta(x)))

            # Create a placeholder column 'session_ID' to make it compatible with the pipeline
            if 'control' in self.log_name:
                df['session_ID'] = df['control_idx']
            else:
                df['session_ID'] = 0




        # save df for potential downstream use. columns=[METRIC_ID, METRIC_NAME, PAT_ID, ACCESS_TIME, ACCESS_INSTANT, USER_ID]
        self.df = df
        # Keep only the necessary columns
        # At this point, we should have : USER_ID, time_delta, session_ID, PAT_ID,  METRIC_NAME
        # (in this order; from least uncertain to most from the author's understanding of the data)
        cols_to_tokenize = [self.prov_col] + ['session_ID'] + [self.pat_col] + self.event_type_cols
        df = df[cols_to_tokenize]


        # Fill all blank cells with a string 'NULL'
        # df.fillna({'PAT_ID':'NULL'}, inplace=True)
        df.loc[:, 'PAT_ID'] = df['PAT_ID'].fillna('NULL')

        # Rename the METRIC_NAME column to ACTION_NAME
        df = df.rename(columns={self.event_type_cols[0]:'ACTION_NAME'})
        self.event_type_cols = ['ACTION_NAME']



        def sessions_to_str(df: DataFrame, cols_to_keep: list, config):
            """
            Combine all rowStrings that are in the same session, using the session_ID as a key. Use
            semicolon+whitespace ("\n ") as a delimiter between rowStrings, and a period as the EOS indicator denoting
            the end of each session.

            :param df:
            # :return: sessionStrings (list)
            """
            list_rowStrings = []
            list_sessionStrings = []
            curr_session_ID = None
            first_action = False

            use_delimiter = config.get('use_delimiter', True)
            if use_delimiter:
                field_delimiter = ""
                row_delimiter = "<ROW>"
            else:
                field_delimiter = ""
                row_delimiter = ""

            # audit_log_headers = field_delimiter.join(cols_to_keep)
            td_cutoff = config.get('timedelta_cutoff', None)
            for i, row in df.iterrows():
                if row['session_ID'] != curr_session_ID:
                    sessionString = ""
                    # sessionString = audit_log_headers + row_delimiter
                    curr_session_ID = row['session_ID']
                    first_action = True

                if (
                    td_cutoff is None
                    or first_action
                    or (td_cutoff is not None
                        and pd.notnull(row['TIME_DELTA'])
                        and row['TIME_DELTA'] >= td_cutoff)
                ):
                    if config['custom_tokenization']:
                        def format_col(col, val):
                            if col == 'ACTION_NAME':
                                return self.action_token_map.get(val, "[ACT_RARE]")
                            elif col == 'TIME_DELTA':
                                if pd.isnull(val):
                                    return '<FIRST_ROW>'
                                elif val <= 2:
                                    return '[TD_0]'
                                elif (val > 2) and (val <= 10):
                                    return '[TD_10]'
                                elif (val > 10) and (val <= 60):
                                    return '[TD_60]'
                                elif val > 60:
                                    return '[TD_>60]'
                            else:
                                return str(val)
                        rowString = field_delimiter.join([
                            format_col(col, row[col]) for col in cols_to_keep
                        ])
                        # rowString = field_delimiter.join([
                        #     str(row[col]) if (col not in ['ACTION_NAME', 'TIME_DELTA'])
                        #     else self.action_token_map.get(row[col], "[ACT_RARE]")
                        #     for col in cols_to_keep
                        # ])
                    else:
                        rowString = field_delimiter.join([str(row[col]) for col in cols_to_keep])

                    first_action = False
                    # list_rowStrings.append(rowString)

                    sessionString += rowString + row_delimiter

                try:
                    if (i == len(df) - 1) or (i < len(df) - 1 and df.loc[i + 1, 'session_ID'] != curr_session_ID):
                        # if current row == last row in session
                        list_sessionStrings.append(sessionString)
                except Exception as e:
                    # Print the error trace
                    print(f"An error occurred: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    print(f"i: {i}, len(df): {len(df)}")
            # return list_rowStrings, list_sessionStrings
            return list_sessionStrings

        if self.num_fields==1:
            cols_to_keep = self.event_type_cols
        else:
            logging.warn("Specify correct num_fields parameter in the config.")

        # print(f"[DEBUG] {self.log_name}: FINAL df shape before sessions_to_str = {df.shape} columns = {df.columns.tolist()}")
        # print(f"[DEBUG] {self.log_name}: HEAD\n{df.head()}")
        # self.rowStrings, self.sessionStrings = sessions_to_str(df, cols_to_keep, config=self.config)
        self.sessionStrings = sessions_to_str(df, cols_to_keep, config=self.config)


    def load_from_cache(self):
        """
        Load the dataset from a cached file.
        :type tokens: bool
        """
        cache_path = os.path.normpath(self.root_dir)
        if not os.path.exists(cache_path):
            raise ValueError("Cache does not exist.")
        if self.unit_string == 'row':
            with open(os.path.normpath(os.path.join(cache_path, "rowStrings.pkl")), "rb") as f:
                try:
                    self.rowStrings = pickle.load(f)
                except EOFError: # if empty file
                    self.rowStrings = None
        else:
            with open(os.path.normpath(os.path.join(cache_path, "sessionStrings.pkl")), "rb") as f:
                try:
                    self.sessionStrings = pickle.load(f)
                except EOFError: # if empty file
                    self.sessionStrings = None


class TokenizedDataSet(Dataset):
    def __init__(self, tokenized_dataset: list):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx].copy()
        result = {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['labels']
        }

        return result
