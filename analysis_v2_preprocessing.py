"""
This script contains the code migrated from "analysis_v2_model.ipynb". It is aimed to tidy up the notebook by removing some preprocessing parts.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import re
from pprint import pprint

# Load the data
df_trans = pd.read_csv('data.v1.csv')
df_code = pd.read_csv('code_snapshot_records.csv')
df_tasklog = pd.read_csv('task_log_records.csv')

# Correct the data type of timestamp
df_trans['timestamp'] = pd.to_datetime(df_trans['timestamp'])
df_code['timestamp'] = pd.to_datetime(df_code['timestamp'])
df_tasklog['timestamp'] = pd.to_datetime(df_tasklog['timestamp'])


# ============================================================
# Preprocess the task log
df_tasklog.sort_values('timestamp', inplace=True)


# ============================================================
# Preprocess the code

# Replace code linebreaks by whitespace
df_code['snapshot'] = df_code['snapshot'].str.replace('\n', ' ')

df_code_postedit = df_code[df_code['category'] == 'PostEdit']  # Fine-grained
df_code_preexec = df_code[df_code['category'] == 'PreExec']

# Pair and join the code

def pair_code_snapshots(code):
    x = code.copy()
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
        p2_code = j.split('... ;')[1:]
        p_merged_code = []
        for k, l in zip(p1_code, p2_code):
            p_merged_code.append(k)
            p_merged_code.append(l)
        merged_code.append(''.join(p_merged_code))

    return p1_id, merged_code, paired_code

df_mc_preexec = []
for _, df_code_gp in df_code_preexec.groupby('group_id'):
    p1_id, sample_mc, sample_pc = pair_code_snapshots(df_code_gp)
    df_code_gp1 = df_code_gp[df_code_gp['user_id'] == p1_id].copy().reset_index()
    df_code_gp1['snapshot'] = sample_mc
    df_mc_preexec.append(df_code_gp1)
df_mc_preexec = pd.concat(df_mc_preexec).reset_index().drop(['level_0', 'index'], axis=1)


# ============================================================
# Preprocess the transcriptions

# Fill empty txt
df_trans['txt'] = df_trans['txt'].fillna('')

# Remove non-task records
df_trans = df_trans[df_trans['task'] != 'X'].copy()

# Append some new features
df_trans['vis_type'] = df_trans['task'].str.get(1)

# Convert timestamp to time offset (grouped by vis_type)
df_split1 = [x for _, x in df_trans.groupby(['group_id', 'vis_type'])]
df_split2 = []
for i, x in enumerate(df_split1):
    x = x.sort_values('timestamp')
    x['time_offset_vt'] = x['timestamp'].diff().fillna(pd.Timedelta(seconds=0)).cumsum()
    x['time_offset_vt'] = x['time_offset_vt'].apply(lambda x: x.seconds)
    x['time_offset_vt'] = x['time_offset_vt'] / x['time_offset_vt'].max()
    df_split2.append(x)
df_trans1 = pd.concat(df_split2)

# Drop useless or redundant columns
df_trans1.drop([
    'group', 'short_id', 'user', 'role', 'txt_path',  'audio_path',
    'time_offset', 'time_cumsum', 'log_path',
], axis=1, inplace=True)


# ============================================================
# Identify collaboration windows
"""
Identify intense collaboration windows.

Rule:
    --- EXEC --- ERROR --- EXEC --- ...
    WORK     WAIT      WORK

Discussion:
    EXEC-RESET: can be WORK or WAIT
    EXEC-ERROR: WAIT
    EXEC-EXEC: ?

    RESET-EXEC: right before exec or after error
    RESET-ERROR: ?
    RESET-RESET: ?

    ERROR-EXEC: WORK
    ERROR-RESET: ?
    ERROR-ERROR: ?

    -EXEC: WORK
    ERROR-: WORK

"""

LOG_STATE_ERROR = 'ERROR'
LOG_STATE_EXEC = 'EXEC'
LOG_STATE_RESET = 'RESET'
LOG_STATE_REWARD = 'REWARD'

WIN_EVENT_EXEC_ERR = 'EXEC-ERR'
WIN_EVENT_ERR_EXEC = 'ERR-EXEC'

# Look for starting times of states

exec_stimes = df_tasklog[
    (df_tasklog['log_details'] == 'Start running the loaded script') |
    (df_tasklog['log_details'] == 'The avater is reset.') |
    (df_tasklog['log_details'] == 'Trap triggered by HeliAvatar') |
    (df_tasklog['log_details'] == 'Trap triggered by HeliAvatarGhost') |
    (df_tasklog['log_details'] == 'A reward is collected.')
].copy()

# Map states
state_mapping = {
    'Trap triggered by HeliAvatar': LOG_STATE_ERROR,
    'Trap triggered by HeliAvatarGhost': LOG_STATE_ERROR,
    'Start running the loaded script': LOG_STATE_EXEC,
    'The avater is reset.': LOG_STATE_RESET,
    'A reward is collected.': LOG_STATE_REWARD,
}
exec_stimes['state'] = df_tasklog['log_details'].apply(lambda x: state_mapping.get(x, 'Undefined'))


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
exec_stimes = exec_stimes.drop(['log_class', 'log_tag', 'log_details', 'group_id'], axis=1)

# Iterate states and identify event windows

# Seperate the exec_stimes table by user_id
tmp = [x for _, x in exec_stimes.groupby('user_id')]
event_windows = []
# For each user x's exec_stimes, extract the event windows
for x in tmp:
    current_state = None
    for i, row in x.iterrows():
        # --- EXEC
        if current_state is None and row['state'] == LOG_STATE_EXEC:
            event_windows.append({
                'user_id': row['user_id'], 'event_win': WIN_EVENT_ERR_EXEC, 'time_until': row['timestamp'], })
            current_state = LOG_STATE_EXEC
        # EXEC --- ERROR
        if current_state == LOG_STATE_EXEC and row['state'] == LOG_STATE_ERROR:
            event_windows.append({
                'user_id': row['user_id'], 'event_win': WIN_EVENT_EXEC_ERR, 'time_until': row['timestamp'], })
            current_state = LOG_STATE_ERROR
        # ERROR --- EXEC
        if current_state == LOG_STATE_ERROR and row['state'] == LOG_STATE_EXEC:
            event_windows.append({
                'user_id': row['user_id'], 'event_win': WIN_EVENT_ERR_EXEC, 'time_until': row['timestamp'], })
            current_state = LOG_STATE_EXEC

    # When reaching the end of states
    # EXEC ---
    if current_state == LOG_STATE_EXEC:
        event_windows.append({
            'user_id': row['user_id'], 'event_win': WIN_EVENT_EXEC_ERR, 'time_until': row['timestamp'], })
    # ERROR ---
    elif current_state == LOG_STATE_ERROR:
        event_windows.append({
            'user_id': row['user_id'], 'event_win': WIN_EVENT_ERR_EXEC, 'time_until': row['timestamp'], })

# Convert to dataframe and set a new index
event_windows = pd.DataFrame(event_windows)

# Annotate transcription logs by event windows
trans_event = []
for i, trans in df_trans1.iterrows():
    user_id = trans['user_id']
    timestamp = trans['timestamp']
    user_event_wins = event_windows[event_windows['user_id'] == user_id]
    event = user_event_wins.loc[(timestamp < user_event_wins['time_until']).idxmax()]['event_win']
    trans_event.append(event)
df_trans1['event_win'] = trans_event


# ============================================================
"""
Export the processed data for later user

state_stimes.csv:

    A timetalbe of identified states based on the task log. The data are recognized by the "user_id" because records on the same device does not incur the time sync problem.

    This is an intermediate file and you probably don't need this.

event_windows.csv:

    A timetable containing event windows by the rule described above.

    This is an intermediate file and you probably don't need this.

merged_code_preexec.csv:

    A timetable of all pre-exec code for groups (group_id). Since the code has been merged (i.e., code on two devices are merged and aligned), the data are recognized by group_id. The user_id is actually irrelevant in this table.

trans_records_v0.csv:

    This file appends event windows to the transaction records coming along with some preprocessed things.


"""

exec_stimes.to_csv('state_stimes.csv', index=None)
df_mc_preexec.to_csv('merged_code_preexec.csv', index=None)
event_windows.to_csv('event_windows.csv', index=None)
df_trans1.to_csv('trans_records_v0.csv', index=None)
