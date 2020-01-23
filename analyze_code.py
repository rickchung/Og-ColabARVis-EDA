from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class CodeInterpreter():
    """
    CodeInterpreter processes records of code edits and computer some relevant statistics.

    Statistics
    - ED (edit distance)
    - DTW (dynamic time wrapping)

    Member DataFrames (labeled by timestamps)

        Solution
        - ed_reg_sol (ED to the regular solution)
        - dtw_abs_sol (DTW to the "abstract" solution)
        - ed_reg_sol_[playerN, TaskN]
        - dtw_abs_sol_[playerN, TaskN]

        General
        - ed_prev (ED to the previous whole script)
        - ed_prev_taskN (ED to the previous snapshot of a task N)
        - ed_prev_playerN (ED to the previous snapshot of the player N)

        Kinds of Edits
        - ed_cmd
        - ed_param

    """

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

        # Build the code-changing sequence
        self.task_edit_history = {
            'A': self._track_edit_history(vis_type='A'),
            'N': self._track_edit_history(vis_type='N'),
        }

        # Add code tokens
        self._add_tokenized_code()

    def _track_edit_history(self, vis_type):
        """
        Build the history of code snapshots by using the initial code and following the post-edit code records.
        """
        # Refer to some data
        init_code = self.init_code
        df_edcode = self.get_postedit_code()

        # Copy the code-edit records as the base
        code_history = df_edcode[['timestamp', 'snapshot']].copy()
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
        Tokenize the code in each state.
        """
        task_edit_history = self.task_edit_history
        tokenize_code = self._tokenize_code
        for vis_type in ['A', 'N']:
            task_edit_history[vis_type]['_state_p1'] = \
                task_edit_history[vis_type]['state_p1'].apply(tokenize_code)
            task_edit_history[vis_type]['_state_p2'] = \
                task_edit_history[vis_type]['state_p2'].apply(tokenize_code)

    def _tokenize_code(self, snapshot):
        """
        Tokenize the code snapshot and remove heading/trailing whitespace.
        """
        tokens = [i.strip() for i in snapshot.split('; ') if len(i) != 0]
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

    def get_task_edit_history(self):
        return self.task_edit_history

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
    # code_snapshots = collect_code_snapshots()
    # task_log_records = collect_task_logdata()

    # Load data

    cols = ['group_id', 'user_id', 'vis_type',
            'timestamp', 'category', 'snapshot']
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

    # For each group, make a code interpreter

    gp_excode = df_code_preexec.query('(group_id == 5)')
    gp_edcode = df_code_postedit.query('(group_id == 5)')

    ci = CodeInterpreter(gp_excode, gp_edcode)
    df_edit_history = ci.get_task_edit_history()

    a = ci.task_edit_history['A'].loc[-1, 'state_p1']
    b = ci.task_edit_history['A'].loc[-1, 'state_p2']
    merged_code = ci._merge_code(a, b)
