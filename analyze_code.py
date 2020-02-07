import difflib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from pprint import pprint

# Edit distance package
# https://github.com/luozhouyang/python-string-similarity#levenshtein
from strsimpy.levenshtein import Levenshtein
from strsimpy.metric_lcs import MetricLCS
from strsimpy.longest_common_subsequence import LongestCommonSubsequence

# Dynamic time wrapping
# https://github.com/pierre-rouanet/dtw
from dtw import dtw as dtw_func


class CodeInterpreter():
    """
    CodeInterpreter processes records of code edits and computer some relevant statistics.

    Statistics
        Metrics
        - ED (edit distance)
        - DTW (dynamic time wrapping)
        - LCS (Longest common subsequence)
        - MLCS (Metric Longest common subsequence)

        Solution
        - ed_reg_sol (dist to the regular solution)
        - ed_abs_sol (dist to the "abstract" solution)
        - ed_reg_sol_[playerN, TaskN]
        - ed_abs_sol_[playerN, TaskN]

        General
        - ed_prev (dist to the previous whole script)
        - ed_prev_taskN (dist to the previous snapshot of a task N)
        - ed_prev_playerN (dist to the previous snapshot of the player N)

        Kinds of Edits
        - ed_cmd
        - ed_param

    """

    # Add initial code generated when the app starts
    p1_initcode_task1 = [
        "Turn (left);",
        "Move (forward);",
        "... ;",
        "Hover ();",
        "Continue_Sec (1);",
        "Hover ();",
        "Continue_Sec (1);",
        "... ;",
        "Move (forward);",
    ]
    p1_initcode_task2 = [
        "Engine (start);",
        "Move (forward);",
        "... ;",
        "Move (forward);",
        "Turn (left);",
        "Turn (left);",
        "Continue_Sec (1);",
        "... ;",
        "Move (forward);",
    ]
    p2_initcode_task1 = [
        "... ;",
        "Move (forward);",
        "Hover ();",
        "Continue_Sec (1);",
        "... ;",
        "Hover ();",
        "Continue_Sec (1);",
        "Move (forward);",
    ]
    p2_initcode_task2 = [
        "... ;",
        "Engine (start);",
        "Move (forward);",
        "Move (forward);",
        "... ;",
        "Turn (left);",
        "Turn (left);",
        "Continue_Sec (1);",
    ]
    # Add regular solutions
    reg_sol = [  # Referring to merged code
        "Engine (start)",
        "Climb (up)",
        "Move (forward)",
        "Continue_Sec (8)",
        "Hover ()",
        "Climb (down)",
        "Continue_Sec (6)",
        "Climb (up)",
        "Continue_Sec (3)",
        "Move (forward)",
        "Continue_Sec (7)",
        "Hover ()",
        "Climb (down)",
    ]

    reg_sol_player1 = [  # Referring to snapshots
        "Engine (start)",
        "Climb (up)",
        "...",
        "Climb (down)",
        "Continue_Sec (6)",
        "Climb (up)",
        "Continue_Sec (3)",
        "...",
        "Climb (down)",
    ]
    reg_sol_player2 = [  # Referring to snapshots
        "...",
        "Move (forward)",
        "Continue_Sec (8)",
        "Hover ()",
        "...",
        "Move (forward)",
        "Continue_Sec (7)",
        "Hover ()",
    ]

    # Add abstract solutions
    abs_sol = [  # Referring to merged code
        "Engine (start)",
        "Climb (up)",
        "Move (forward)",
        "Climb (down)",
        "Climb (up)",
        "Move (forward)",
        "Climb (down)",
    ]
    abs_sol_ncp = [
        "Engine (start)",
        "Climb (up)",
        "Move (forward)",
        "Continue_Sec",
        "Climb (down)",
        "Continue_Sec",
        "Climb (up)",
        "Continue_Sec",
        "Move (forward)",
        "Continue_Sec",
        "Climb (down)",
    ]

    def __init__(self, df_excode, df_edcode):
        """
        The input dataframes must be data from the same group.
        """
        # Copy the input dataframes
        self.df_excode = df_excode.copy()
        self.df_edcode = df_edcode.copy()

        # Identify the task order
        t0_ar = gp_excode.xs('A', level=1).iloc[0]['timestamp']
        t0_nonar = gp_excode.xs('N', level=1).iloc[0]['timestamp']
        if t0_ar > t0_nonar:
            self.task1_vis, self.task2_vis = 'N', 'A'
        else:
            self.task1_vis, self.task2_vis = 'A', 'N'
        # Get the initial code for the two tasks
        self.init_code = self._get_init_code()

        # Identify roles in post-edit code
        self.df_edcode['role'] = \
            self.df_edcode['snapshot'].apply(self._identify_player)
        # Reset the index
        self.df_edcode = \
            self.df_edcode.set_index(['vis_type', 'role']).sort_index()

        # Track code states
        self.task_edit_history = {
            'A': self._track_code_states(vis_type='A'),
            'N': self._track_code_states(vis_type='N'),
        }
        # Add code tokens
        self._add_tokenized_code()

        # Measure ED to the regular solution
        self._add_dist_to_ref(self.reg_sol, '_state', 'ed_reg_sol')

        # Measure ED to the abstract solution
        self._add_dist_to_ref(self.abs_sol, '_state', 'ed_abs_sol')
        self._add_dist_to_ref(self.abs_sol_ncp, '_state_ncp', 'ed_abs_sol_ncp')

        # Measure ED to regular solutions of task1 and task2
        self._add_dist_to_ref(self.reg_sol[:7],
                              '_state_task1', 'ed_reg_sol_task1')
        self._add_dist_to_ref(self.reg_sol[7:],
                              '_state_task2', 'ed_reg_sol_task2')
        # Measure ED to abstract solutions of task1 ans task2
        self._add_dist_to_ref(self.abs_sol[:4],
                              '_state_task1', 'ed_abs_sol_task1')
        self._add_dist_to_ref(self.abs_sol[4:],
                              '_state_task2', 'ed_abs_sol_task2')
        self._add_dist_to_ref(self.abs_sol_ncp[:5],
                              '_state_task1', 'ed_abs_sol_task1_ncp')
        self._add_dist_to_ref(self.abs_sol_ncp[5:],
                              '_state_task2', 'ed_abs_sol_task2_ncp')

        # Measure ED to regular solutions of player1 and player2
        self._add_dist_to_ref(self.reg_sol_player1,
                              '_state_p1', 'ed_reg_sol_p1')
        self._add_dist_to_ref(self.reg_sol_player2,
                              '_state_p2', 'ed_reg_sol_p2')

        # Find exact code differences between code states
        # e.g., (L1, removed, added), (L3, removed, added), ...
        self._add_code_diff()

    # Methods for internal use only

    def _track_code_states(self, vis_type):
        """
        Build the history of code snapshots (code states) by using the initial code and following the post-edit code records.
        """
        # Refer to some data
        init_code = self.init_code
        df_edcode = self.get_postedit_code()

        # Copy the code-edit records as the base
        # Here only a minimum set of columns are kept.
        code_history = df_edcode[['timestamp',
                                  'time_offset_vt', 'snapshot']].copy()
        task_history = code_history.loc[vis_type].copy()
        task_history.sort_values('timestamp', inplace=True)

        # Insert new ordered indices
        task_history.reset_index(inplace=True)

        # Add one row for the initial code and then sort the df
        task_history.loc[-1] = None
        task_history.sort_index(inplace=True)

        # Insert initial code as the default state
        task_history['state_p1'] = ' '.join(init_code[vis_type]['P1'])
        task_history['state_p2'] = ' '.join(init_code[vis_type]['P2'])

        # Update code states by iterating code-edit records
        states = []
        prev_s1, prev_s2 = np.nan, np.nan
        for i, (index, row) in enumerate(task_history.iterrows()):
            role = row['role']
            snapshot = row['snapshot']
            state_p1, state_p2 = row['state_p1'], row['state_p2']
            rt = (np.nan, np.nan)

            # When role is NaN, it should be the initial state
            if role is np.nan:
                rt = (state_p1, state_p2)

            # For an edit, keep the state from the previous row
            else:
                if prev_s1 is np.nan or prev_s2 is np.nan:
                    raise ValueError('the previous state was not found')
                # Save the edit record from P1 or P2
                if role == 'P1':
                    rt = (snapshot, prev_s2)
                elif role == 'P2':
                    rt = (prev_s1, snapshot)
                else:
                    raise ValueError('Unknown role value: {}'.format(role))

            prev_s1, prev_s2 = rt
            states.append(rt)

        # Add back the code states to the base df
        # Note: The index and column names have to match in this case
        states = pd.DataFrame(states, columns=['state_p1', 'state_p2', ])
        task_history['state_p1'] = states['state_p1'].values
        task_history['state_p2'] = states['state_p2'].values

        task_history['state'] = \
            task_history.apply(
                lambda x: self._merge_code(x['state_p1'], x['state_p2']), axis=1)

        return task_history

    def _get_init_code(self):
        """
        Get initial code snapshots for P1 and P2 in the two tasks.
        """
        task1_vis, task2_vis = self.task1_vis, self.task2_vis
        init_code = {
            'A': {
                'P1': None,
                'P2': None,
            },
            'N': {
                'P1': None,
                'P2': None,
            },
        }
        init_code[task1_vis]['P1'] = self.p1_initcode_task1
        init_code[task2_vis]['P1'] = self.p1_initcode_task2
        init_code[task1_vis]['P2'] = self.p2_initcode_task1
        init_code[task2_vis]['P2'] = self.p2_initcode_task2

        return init_code

    def _identify_player(self, code):
        """
        Identify the role of player by checking the signature in the code string.
        """
        return 'P2' if code[:3] == '...' else 'P1'

    def _add_tokenized_code(self):
        """
        Tokenize the code in each state and create different clips of code.
        """
        task_edit_history = self.task_edit_history
        tokenize_code = self._tokenize_code
        for vis_type in ['A', 'N']:
            # Player 1 and 2
            task_edit_history[vis_type]['_state_p1'] = \
                task_edit_history[vis_type]['state_p1'].apply(tokenize_code)
            task_edit_history[vis_type]['_state_p2'] = \
                task_edit_history[vis_type]['state_p2'].apply(tokenize_code)

            # General state
            task_edit_history[vis_type]['_state'] = \
                task_edit_history[vis_type]['state'].apply(tokenize_code)
            # Task1 and Task2
            task_edit_history[vis_type]['_state_task1'] = \
                task_edit_history[vis_type]['_state'].apply(lambda x: x[:7])
            task_edit_history[vis_type]['_state_task2'] = \
                task_edit_history[vis_type]['_state'].apply(lambda x: x[7:])

            # no "continue" parameters
            task_edit_history[vis_type]['_state_ncp'] = \
                task_edit_history[vis_type]['state'].apply(
                    tokenize_code, remove_continue_params=True)
            task_edit_history[vis_type]['_state_ncp_task1'] = \
                task_edit_history[vis_type]['_state_ncp'].apply(
                    lambda x: x[:7])
            task_edit_history[vis_type]['_state_ncp_task2'] = \
                task_edit_history[vis_type]['_state_ncp'].apply(
                    lambda x: x[7:])

    def _add_dist_to_ref(self, ref_seq, target_col, out_col_name, metric='mylcs'):
        """
        Measure distance between the given ref_seq to all values in the column target_col of task_edit_history.
        """
        for vis_type in ['A', 'N']:
            df = self.task_edit_history[vis_type]
            ed = df[target_col].apply(
                lambda x: self._get_distance(x, ref_seq, metric=metric,
                                             max_len=(len(x) + len(ref_seq))))
            self.task_edit_history[vis_type][out_col_name] = ed

    def _add_code_diff(self):
        """
        Find the changes between snapshots
        """
        for vis_type in ['A', 'N']:
            df = self.task_edit_history[vis_type]
            states = df['_state']
            shifted_states = df['_state'].shift()
            changes = [
                self._get_diff(p, c) for p, c in zip(shifted_states, states)]
            self.task_edit_history[vis_type]['state_diff'] = changes

    # Helper methods

    def _tokenize_code(self, snapshot, remove_continue_params=False):
        """
        Tokenize the code snapshot and remove heading/trailing whitespace.
        """
        tokens = [i.strip()
                  for i in snapshot.split(';') if len(i.strip()) != 0]

        if remove_continue_params:
            tokens_ncp = []
            for i in tokens:
                if 'Continue_Sec' in i:
                    tokens_ncp.append(i.split('(')[0].strip())
                else:
                    tokens_ncp.append(i)
            tokens = tokens_ncp

        return tokens

    def _merge_code(self, p1_code_str, p2_code_str):
        """
        Merge two code strings by substituting '...' blocks
        """
        merged_code = []

        # Split P1's code
        p1_code = p1_code_str.split('... ;')
        # Split, remove the heading empty line, and append one empty item for alignment
        p2_code = p2_code_str.split('... ;')[1:] + ['']

        p_merged_code = []
        for k, l in zip(p1_code, p2_code):
            p_merged_code.append(k)
            p_merged_code.append(l)
        merged_code = ''.join(p_merged_code)

        return merged_code

    def _get_distance(self, a, b, metric='ed', max_len=None):
        """
        Compute the edit distance between two token lists a and b.
        """
        rt = np.nan
        if metric == 'ed':
            lev = Levenshtein()
            rt = lev.distance(a, b)
        elif metric == 'dtw':
            def dist_func(x, y): return 0 if x == y else 1
            d, mat_cost, mat_acc_cost, path = dtw_func(a, b, dist=dist_func)
            rt = d
        elif metric == 'lcs':
            lcs = LongestCommonSubsequence()
            rt = lcs.distance(a, b)
        elif metric == 'mylcs':
            lcs = LongestCommonSubsequence()
            rt = lcs.distance(a, b) / max_len
        elif metric == 'mlcs':  # metric LCS
            metric_lcs = MetricLCS()
            rt = metric_lcs.distance(a, b)
        else:
            raise NotImplementedError(
                "Metric not implemented: {}".format(metric))

        return rt

    def _get_diff(self, a, b):
        """
        Find the difference between two token lists a and b.
        """
        if a is np.nan or b is np.nan:
            return np.nan
        diff = list(difflib.unified_diff(a, b, n=0))
        rt = '|'.join(diff)

        # rt = []
        # for i in diff:
        #     if i[:3] == '---' or i[:3] == '+++':
        #         pass
        #     elif i[:2] == '@@':
        #         rt.append(i.split('@@')[1].strip())
        #     elif i[0] == '-' or i[0] == '+':
        #         rt.append(i)
        # rt = '|'.join(rt)

        return rt

    # Getter methods

    def get_task_edit_history(self):
        # Merge AR and NonAR dataframes
        df_ar = self.task_edit_history['A'].copy()
        df_ar['vis_type'] = 'AR'
        df_nonar = self.task_edit_history['N'].copy()
        df_nonar['vis_type'] = 'Non-AR'
        rt = pd.concat([df_ar, df_nonar], ignore_index=True, axis='index')
        # Remove columns starting with '_'
        cols = [c for c in rt.columns if not c.startswith('_')]
        return rt[cols].copy()

    def get_preexec_code(self):
        """
        Get records of pre-exec codes.
        """
        return self.df_excode

    def get_postedit_code(self):
        """
        Get records of post-edit codes.
        """
        return self.df_edcode


if __name__ == "__main__":
    # Load data

    cols = ['group_id', 'user_id', 'vis_type',
            'timestamp', 'time_offset_vt', 'category', 'snapshot']
    df_code_preexec = pd.read_csv('code_preexec.csv')[cols].copy()
    df_code_postedit = pd.read_csv('code_postedit.csv')[cols].copy()

    # Preprocess

    def convert_timestamp(x): return pd.to_datetime(x)
    df_code_preexec['timestamp'] = convert_timestamp(
        df_code_preexec['timestamp'])
    df_code_postedit['timestamp'] = convert_timestamp(
        df_code_postedit['timestamp'])

    def get_group_index(x): return int(
        x.split('-')[0].replace('UserStudy', ''))
    df_code_preexec['group_id'] = df_code_preexec['group_id'].apply(
        get_group_index)
    df_code_postedit['group_id'] = df_code_postedit['group_id'].apply(
        get_group_index)

    def sort_by_time(x): return x.sort_values('timestamp')
    df_code_preexec = df_code_preexec.groupby(
        ['group_id', 'vis_type']).apply(sort_by_time)
    df_code_postedit = df_code_postedit.groupby(
        ['group_id', 'vis_type']).apply(sort_by_time)

    # Interpret the code group by group
    rt = []
    for gid in range(2, 14):
        gp_excode = df_code_preexec.query('(group_id == {})'.format(gid))
        gp_edcode = df_code_postedit.query('(group_id == {})'.format(gid))
        ci = CodeInterpreter(gp_excode, gp_edcode)
        df_edit_history = ci.get_task_edit_history()
        df_edit_history['gid'] = gid
        rt.append(df_edit_history)
    rt = pd.concat(rt, axis='index', ignore_index=True)
    rt.to_csv('code_stats.csv', index=False)
