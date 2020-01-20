"""
This script contains the code migrated from "analysis_v2_model.ipynb". It is aimed to tidy up the notebook by removing some preprocessing parts.

Output Files

    state_stimes.csv:

        A timetalbe of identified states based on the task log. The data are recognized by the "user_id" because records on the same device does not incur the time sync problem.

        This is an intermediate file and you probably don't need this.

    event_windows.csv:

        A timetable containing event windows by the rule described above.

        This is an intermediate file and you probably don't need this.

    code_preexec.csv:

        A timetable of all pre-exec code for groups (group_id). Since the code has been merged (i.e., code on two devices are merged and aligned), the data are recognized by group_id. The user_id is actually irrelevant in this table.

    code_postedit.csv

        A time table of all post-edit code snapshots. This table should be interpreted on a user-by-user basis because there may be only one user modified his/her code at times. The code snapshots of P1 and P2 are not merged because it is not easy to align code by following the current version of task log.

    trans_records_v0.csv:

        This file appends event windows to the transaction records coming along with some preprocessed things.


"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
import re
import difflib
from collections import Counter
from pprint import pprint

# Edit distance package
# https://github.com/luozhouyang/python-string-similarity#levenshtein
from strsimpy.levenshtein import Levenshtein as Lev


def pair_code_snapshots(dfgp_code):
    """
    Pair and merge code in the dataframe dfgp_code.

    Returns:
        - The user ID of the first player
        - A series of merged code (items are strings)
        - A series of paired code (items are 2-item tuples)
    """
    x = dfgp_code.copy()
    x.sort_values('timestamp', inplace=True)
    (u1_id, u1_code), (u2_id, u2_code) = x.groupby('user_id')

    # Find the P1
    re_p2 = r'^\.\.\. ;'
    if re.match(re_p2, u1_code['snapshot'].iloc[0]):
        p1_id = u2_id
    else:
        p1_id = u1_id

    # Pair the code
    paired_code = []
    merged_code = []
    for i, j in zip(u1_code['snapshot'], u2_code['snapshot']):
        if re.match(re_p2, i):
            i, j = j, i
        paired_code.append((i, j))

        p1_code = i.split('... ;')
        # Split, remove the heading empty line, and append one empty item for alignment
        p2_code = j.split('... ;')[1:] + ['']
        p_merged_code = []
        for k, l in zip(p1_code, p2_code):
            p_merged_code.append(k)
            p_merged_code.append(l)
        merged_code.append(''.join(p_merged_code))

    return p1_id, merged_code, paired_code


def gen_event_win_record(row, event_win):
    """
    A helper function for generating a record of an event window.
    """
    return {
        'user_id': row['user_id'],
        'event_win': event_win,
        'time_until': row['timestamp'],
    }


def annotate_event_wins(df, event_windows, heuristic=True):
    """
    Iterate df and annotate event windows on rows by using timestamps in event_windows.
    """
    trans_event = []
    for i, trans in df.iterrows():
        user_id = trans['user_id']
        timestamp = trans['timestamp']
        user_event_wins = event_windows[event_windows['user_id'] == user_id]

        # Try to find the event window covers the most time of transcription clip
        if heuristic:
            audio_len = trans['audio_len']
            # Find candidate events and use the one covering the most talk time
            e1e2 = user_event_wins[timestamp <
                                   user_event_wins['time_until']].iloc[:2]
            # If a trans goes over the boundary, use the last window
            if e1e2.shape[0] == 0:
                event = user_event_wins.iloc[-1]['event_win']
            # If there's only one matched window, use it
            elif e1e2.shape[0] != 2:
                event = e1e2.iloc[0]['event_win']
            else:
                e1, e2 = e1e2.iloc[0], e1e2.iloc[1]
                e1_coverage = e1['time_until'] - timestamp
                e2_coverage = e2['time_until'] - timestamp
                # If e2 goes beyond the coverage of trans, use e1
                if e2_coverage.seconds > audio_len:
                    event = e1['event_win']
                # If e2 covers more than e1, use e2
                elif (e2_coverage - e1_coverage).seconds > e1_coverage.seconds:
                    event = e2['event_win']
                else:
                    event = e1['event_win']
        # Just find the closest match
        else:
            e1 = user_event_wins[
                timestamp < user_event_wins['time_until']].iloc[:1]
            if e1.shape[0] == 0:
                event = user_event_wins.iloc[-1]['event_win']
            else:
                event = e1.iloc[0]['event_win']

        trans_event.append(event)
    return trans_event


def _find_code_diff(x):
    """
    Find changes between two code snapshots and count numbers of lines changed.

    Type of code changes:

        1. Tweaking parameters

            - Continue_Sec (4)
            ?               ^
            + Continue_Sec (5)
            ?               ^

        2. Tweaking functions

            - Climb (down)
            + Hover ()
    """
    code_snapshots = x['snapshot']
    def get_code(x): return [c.strip() for c in x.split('; ') if len(c) != 0]
    differ = difflib.Differ()

    shifted = code_snapshots.shift()
    diffs_list = []
    num_removed = []
    num_added = []
    num_arg_tweaked = []
    for s1, s2 in zip(shifted.values, code_snapshots.values):
        if s1 is np.nan or s2 is np.nan:
            diffs_list.append(np.nan)

            num_removed.append(np.nan)
            num_added.append(np.nan)
            num_arg_tweaked.append(np.nan)
        else:
            s1code, s2code = get_code(s1), get_code(s2)
            diffs = [i for i in differ.compare(
                s1code, s2code) if i[:2] != '  ']
            diffs_list.append(diffs)

            c = Counter([i[0] for i in diffs])
            num_removed.append(c['-'])
            num_added.append(c['+'])
            num_arg_tweaked.append(c['?'] / 2)

    rt = x.copy()
    rt['code_diff'] = diffs_list
    rt['num_removed'] = num_removed
    rt['num_added'] = num_added
    rt['num_arg_tweaked'] = num_arg_tweaked

    return rt


def _mark_vis_type_code(x, timetable):
    index_value = x.name
    a_time = timetable.loc[index_value, 'A']
    n_time = timetable.loc[index_value, 'N']

    # Remove all snapshots saved before the first task
    first, second = (a_time, n_time) if a_time <= n_time else (n_time, a_time)
    rt = x[x['timestamp'] >= first].copy()

    # Mark the rest of snapshots
    rt['vis_type'] = 'Unknown'
    if a_time < n_time:
        # If A comes first
        rt.loc[(rt['timestamp'] >= a_time), ['vis_type']] = 'A'
        rt.loc[(rt['timestamp'] >= n_time), ['vis_type']] = 'N'
    else:
        # If N comes first
        rt.loc[(rt['timestamp'] >= n_time), ['vis_type']] = 'N'
        rt.loc[(rt['timestamp'] >= a_time), ['vis_type']] = 'A'

    # Mark time offset
    def _mark_time_offset(y):
        y = y.sort_values('timestamp')
        rt1 = y['timestamp'].diff().fillna(pd.Timedelta(seconds=0)).cumsum()
        rt1 = rt1.apply(lambda x: x.seconds)
        rt1 = rt1 / rt1.max()
        return rt1
    time_offsets = rt.groupby(['vis_type']).apply(_mark_time_offset)
    rt['time_offset_vt'] = None
    rt.loc[rt['vis_type'] == 'A', 'time_offset_vt'] = time_offsets.loc['A']
    rt.loc[rt['vis_type'] == 'N', 'time_offset_vt'] = time_offsets.loc['N']

    return rt


def _get_stage_clear_times(x):
    """
    Returns:
        Stage-clear times of tutorial, task1, task2 in a tuple
    """
    rt = [np.nan, np.nan, np.nan]

    # Find all stage-clear times
    clear_times = x[x['log_details'].str.startswith('Stage Clear')].copy()

    # Merge those times that are too close
    # (it means they are just duplicate records from the two devices)
    clear_times['offset'] = (clear_times['timestamp'].diff()).fillna(
        pd.Timedelta(seconds=3600))
    event_times = clear_times[clear_times['offset']
                              >= pd.Timedelta(seconds=60)]
    for i, item in enumerate(event_times['timestamp'].values):
        rt[i] = item

    return pd.Series(rt, index=['t0_clear_time', 't1_clear_time', 't2_clear_time'])


def _get_relaxed_solution(sol):
    """
    Get an relaxed solution (by removing args in continue sec)
    """
    rt = []
    for i in sol:
        if i == "...":
            rt.append(i)
        else:
            cmd, arg = i.split(' ')
            if cmd == 'Continue_Sec':
                rt.append(cmd)
            else:
                rt.append(i)
    return rt


def _get_editdist_solution(snapshot, solution, solution_r):
    """
    Split code snapshots and calculate edit distance to the solution
    """
    split = [i.strip() for i in snapshot.split('; ') if len(i) != 0]
    split_r = _get_relaxed_solution(split)
    lev = Lev()
    ed_sol = lev.distance(split, solution)
    ed_sol_r = lev.distance(split_r, solution_r)
    return pd.Series({'ed_sol': ed_sol, 'ed_sol_r': ed_sol_r})


# ============================================================
# Load the data
print("[INFO] Loading input files")
df_trans = pd.read_csv('data.v1.csv')
df_code = pd.read_csv('code_snapshot_records.csv')
df_tasklog = pd.read_csv('task_log_records.csv')

# Correct the data type of timestamp
df_trans['timestamp'] = pd.to_datetime(df_trans['timestamp'])
df_code['timestamp'] = pd.to_datetime(df_code['timestamp'])
df_tasklog['timestamp'] = pd.to_datetime(df_tasklog['timestamp'])


# ============================================================
# Preprocess the task log
print("[INFO] Processing task log data")
df_tasklog.sort_values('timestamp', inplace=True)

# Extract the time when a map is opened and cleared
stage_clear_times = df_tasklog[
    (df_tasklog['log_details'].str.startswith('MAP, A map is loaded')) |
    (df_tasklog['log_details'].str.startswith('Stage Clear'))
].copy()
stage_clear_times = stage_clear_times\
    .groupby('group_id')\
    .apply(_get_stage_clear_times)

# ============================================================
# Preprocess the transcriptions
print("[INFO] Processing discourse transcriptions")

# Fill empty txt
df_trans['txt'] = df_trans['txt'].fillna('')

# Remove non-task records
df_trans = df_trans[df_trans['task'] != 'X'].copy()

# Append some new features
df_trans['vis_type'] = df_trans['task'].str.get(1)

# Convert timestamp to time offset (grouped by vis_type and speaker)
df_split1 = [x for _, x in df_trans.groupby(
    ['group_id', 'speaker', 'vis_type'])]
df_split2 = []
for i, x in enumerate(df_split1):
    x = x.sort_values('timestamp')
    x['time_offset_vt'] = x['timestamp'].diff().fillna(
        pd.Timedelta(seconds=0)).cumsum()
    x['time_offset_vt'] = x['time_offset_vt'].apply(lambda x: x.seconds)
    x['time_offset_vt'] = x['time_offset_vt'] / x['time_offset_vt'].max()
    df_split2.append(x)
df_trans1 = pd.concat(df_split2)

# Drop useless or redundant columns
df_trans1.drop(['group', 'time_offset', 'time_cumsum', ], axis=1, inplace=True)


# ============================================================
# Identify collaboration windows
print("[INFO] Processing event windows")

"""
Identify intense collaboration windows.

Rule:
    --- EXEC --- ERROR --- EXEC --- ...
    WORK     WAIT      WORK
"""

LOG_STATE_MAP_INIT = 'MAP_INIT'
LOG_STATE_ERROR = 'ERROR'
LOG_STATE_EXEC = 'EXEC'
LOG_STATE_RESET = 'RESET'
LOG_STATE_REWARD = 'REWARD'

WIN_EVENT_H_ERR = 'ERR---'
WIN_EVENT_T_ERR = '---ERR'
WIN_EVENT_H_EXEC = 'EXE---'
WIN_EVENT_T_EXEC = '---EXE'

# Mark starting times of states in the task log

# Only keep a subset of the task log data
exec_stimes = df_tasklog[
    (df_tasklog['log_details'] == 'Start running the loaded script') |
    (df_tasklog['log_details'] == 'The avater is reset.') |
    (df_tasklog['log_details'] == 'Trap triggered by HeliAvatar') |
    (df_tasklog['log_details'] == 'Trap triggered by HeliAvatarGhost') |
    (df_tasklog['log_details'] == 'A reward is collected.') |
    (df_tasklog['log_details'].str.startswith('MAP'))
].copy()

# Map states to state tokens
state_mapping = {
    'Trap triggered by HeliAvatar': LOG_STATE_ERROR,
    'Trap triggered by HeliAvatarGhost': LOG_STATE_ERROR,
    'Start running the loaded script': LOG_STATE_EXEC,
    'The avater is reset.': LOG_STATE_RESET,
    'A reward is collected.': LOG_STATE_REWARD,
}
exec_stimes['state'] = df_tasklog['log_details'].apply(
    lambda x: state_mapping.get(x, 'Undefined'))
exec_stimes.loc[exec_stimes['log_details'].str.startswith(
    'MAP'), 'state'] = LOG_STATE_MAP_INIT

# Group the df by user_id and calculate the size of windows (duration of states)
tmp1 = [x for _, x in exec_stimes.groupby('user_id')]
tmp2 = []
for x in tmp1:
    x = x.sort_values('timestamp')
    x['duration'] = x['timestamp'].diff(1).fillna(pd.Timedelta(seconds=0))
    x['duration'] = x['duration'].shift(-1)
    x['duration'] = x['duration'].apply(lambda x: x.seconds)
    tmp2.append(x)
exec_stimes = pd.concat(tmp2)
exec_stimes = exec_stimes[exec_stimes['duration'] > 0.0].copy()
exec_stimes = exec_stimes.drop(
    ['log_class', 'log_tag', 'log_details', 'group_id'], axis=1)

# Sometimes students pressed reset when they think the script has some problem.
# For example, when they saw the helicopter did not descend low enough. In such
# a case, no trap would be triggered. To accommodate this case, I treat the
# event "RESET" as a signal of spotting "ERROR". To distinguish a regular reset
# that usually occurred before an execuation, I set the threshold of duration
# as 5 seconds with assumption that students always pressed "run" right after
# "reset" if they intended to run the program but not any other reasons.
exec_stimes.loc[
    (exec_stimes['state'] == LOG_STATE_RESET) &
    (exec_stimes['duration'] > 5.0),
    'state'] = LOG_STATE_ERROR

# Seperate the exec_stimes table by user_id
user_exec_stimes = [x for _, x in exec_stimes.groupby('user_id')]
event_windows = []

# Iterate states and identify event windows
# For each user x's exec_stimes (timeline), annotate event windows
for x in user_exec_stimes:
    prev_event_state = None
    prev_event_window = None

    for i, row in x.iterrows():
        # The first exec
        if prev_event_state is None and row['state'] == LOG_STATE_EXEC:
            prev_event_state = LOG_STATE_EXEC

        else:
            # EXEC --- until this point
            if (prev_event_state == LOG_STATE_EXEC and
                    row['state'] == LOG_STATE_ERROR):

                event_windows.append(
                    gen_event_win_record(row, WIN_EVENT_H_EXEC))
                prev_event_state = LOG_STATE_ERROR
                prev_event_window = WIN_EVENT_H_EXEC

            # ERR --- until this point
            elif (prev_event_state == LOG_STATE_ERROR and
                  row['state'] == LOG_STATE_EXEC):

                event_windows.append(
                    gen_event_win_record(row, WIN_EVENT_H_ERR))
                prev_event_state = LOG_STATE_EXEC
                prev_event_window = WIN_EVENT_H_ERR

    # When reaching the end of states, mark the last event window
    event_windows.append(gen_event_win_record(row, prev_event_window))

# Convert to dataframe and set a new index
event_windows = pd.DataFrame(event_windows)


# ============================================================
# Preprocess the code
print("[INFO] Processing code snapshots")

# Replace code linebreaks by whitespace
df_code['snapshot'] = df_code['snapshot'].str.replace('\n', ' ')

# Seperate two kinds of code snapshots
# - PostEdit: snapshots saved right after an edit
# - PreExec: snapshots saved right before an execution
df_code_postedit = df_code[df_code['category'] == 'PostEdit'].copy()
df_code_preexec = df_code[df_code['category'] == 'PreExec'].copy()

# Pair and join the code (PreExec Only)
df_merged_code = []
for _, dfgp_code in df_code_preexec.groupby('group_id'):
    # Pair and merge code snapshots
    p1_id, merged_code, _ = pair_code_snapshots(dfgp_code)
    # Select the dataframe which belongs to the user
    dfgp_code1 = dfgp_code[dfgp_code['user_id'] == p1_id]
    dfgp_code1 = dfgp_code1.copy().reset_index()
    # Replace the snapshots in the (copied) input dataframe
    dfgp_code1['snapshot'] = merged_code
    # Save for later processing
    df_merged_code.append(dfgp_code1)
# Concatenate all group dataframes
df_merged_code = pd.concat(df_merged_code).reset_index()
df_merged_code = df_merged_code.drop(['level_0', 'index'], axis=1)
# Save for later processing
df_code_preexec = df_merged_code.copy()

# Mark vis_type and remove tutorial code
vis_times = df_trans1.groupby(['group_id', 'vis_type'])['timestamp'].min()
df_code_preexec1 = df_code_preexec.groupby(['group_id'])\
    .apply(_mark_vis_type_code, timetable=vis_times)
df_code_preexec1 = df_code_preexec1.reset_index(drop=True)
df_code_postedit1 = df_code_postedit.groupby(['group_id'])\
    .apply(_mark_vis_type_code, timetable=vis_times)
df_code_postedit1 = df_code_postedit1.reset_index(drop=True)

# Find differences between pre-exec code snapshots
df_code_preexec2 = df_code_preexec1.groupby(['group_id', 'vis_type'])\
    .apply(_find_code_diff).reset_index(drop=True)
df_code_postedit2 = df_code_postedit1.groupby(['user_id', 'vis_type'])\
    .apply(_find_code_diff).reset_index(drop=True)

# Find the records showing code differences
df_cedit = df_code_postedit2[df_code_postedit2['num_removed'] > 0].copy()
df_cexec = df_code_preexec2[df_code_preexec2['num_removed'] > 0].copy()
# Label types of code editing
df_cedit['edit_label'] = 'EDIT_CMD'
df_cedit.loc[df_cedit['num_arg_tweaked'] > 0, 'edit_label'] = 'EDIT_PARAM'
# Group time series by bins
num_bins = 5
df_cedit['timebins'] = pd.cut(
    df_cedit['time_offset_vt'], num_bins,
    labels=['T' + str(i) for i in range(num_bins)])

# Add the solution code (for pre-exec snapshots)
solution1 = [
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
relaxed_solution1 = _get_relaxed_solution(solution1)

# Compute edit distance to the solution (ed_sol)
df_cexec[['ed_sol', 'ed_sol_r']] = df_cexec['snapshot'].apply(
    _get_editdist_solution, solution=solution1, solution_r=relaxed_solution1)

# Add the solution code for post-edit snapshots
pe_solution1 = [
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
pe_solution2 = [
    "...",
    "Move (forward)",
    "Continue_Sec (8)",
    "Hover ()",
    "...",
    "Move (forward)",
    "Continue_Sec (7)",
    "Hover ()",
]
pe_relaxed_solution1 = _get_relaxed_solution(pe_solution1)
pe_relaxed_solution2 = _get_relaxed_solution(pe_solution2)

# Compute edit distance to the solution (ed_sol)
df_cedit['ed_sol'], df_cedit['ed_sol_r'] = np.nan, np.nan
# For P2 code
i_p2code = df_cedit['snapshot'].str.startswith("...")
df_cedit.loc[i_p2code, ['ed_sol', 'ed_sol_r']] = \
    df_cedit.loc[i_p2code, 'snapshot'].apply(
        _get_editdist_solution, solution=pe_solution2, solution_r=pe_relaxed_solution2)
# For P1 code
i_p1code = (~i_p2code)
df_cedit.loc[i_p1code, ['ed_sol', 'ed_sol_r']] = \
    df_cedit.loc[i_p1code, 'snapshot'].apply(
        _get_editdist_solution, solution=pe_solution1, solution_r=pe_relaxed_solution1)

# ============================================================
# Annotate transcription logs by event windows
print('[INFO] Marking event windows')
df_trans1['event_win'] = annotate_event_wins(df_trans1, event_windows)
# Annotate code snapshots by event windows
df_cexec['event_win'] = annotate_event_wins(
    df_cexec, event_windows, heuristic=False)
df_cedit['event_win'] = annotate_event_wins(
    df_cedit, event_windows, heuristic=False)


# ============================================================
# Save processed data
print("[INFO] Exporting processed data")
exec_stimes.to_csv('state_stimes.csv', index=None)
event_windows.to_csv('event_windows.csv', index=None)
df_cexec.to_csv("code_preexec.csv", index=None)
df_cedit.to_csv("code_postedit.csv", index=None)
df_trans1.to_csv('trans_records_v0.csv', index=None)
stage_clear_times.to_csv('stage_clear_times.csv')
