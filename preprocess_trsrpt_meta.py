#!python3

"""
This script preprocesses the audio info table (containing speech-to-text
transcriptions) by appending some metadata of the user study like task orders,
timestamps of audio clips (starting times of recording), and study groups. The
output file will be processed manually by the researcher for later analyses.
"""

import pandas as pd
from utils import gen_audio_info_tab, get_wav_duration, read_task_log

def append_player_roles(input_df, id_col='short_id'):
    """
    Assign player roles to each record in the input_df. The ad-hoc rule was made by myself.
    """
    output_df = input_df.copy()
    output_df['role'] = 'Unknown'

    users = output_df[id_col].unique()
    for u in users:
        user_df = output_df[output_df[id_col] == u]
        user_log_fname = user_df['log_path'].values[0]
        log_file = read_task_log(user_log_fname)
        first_keyword = log_file.query(
            'log_class == "EditorTempCmdWithNumbers"').iloc[0]['log_details']

        # Use the first modification as the keyword for identifying the role
        if 'to Engine' in first_keyword:
            role = 'P1'
        elif 'to Move' in first_keyword:
            role = 'P2'
        else:
            role = 'Unknown'

        # Some exceptions
        if u == 'BC-11222019150347' or u == 'LA-11272019090650':
            role = 'P1'

        # Update the dataframe
        output_df.loc[output_df['short_id'] == u, 'role'] = role

    return output_df

def append_meta_data(tab, trans_type='machine'):
    """
    Append meta data about the task to the table. Some rules are made only for this specific user study. Some columns will be renamed for readability.
    """
    # A map from audio_path to its task (e.g., task1-AR,task1-NonAR, etc.)
    # This file was prepared manually based on the very first version of
    # preprocessed dataset from this script and some duplicate columns were not removed.
    tab_taskmap = pd.read_csv('proc_data/data.timeline.mapping.csv')
    _t1 = tab_taskmap['audio_path']
    _t2 = tab_taskmap['task']
    taskmap = dict(zip(_t1, _t2))

    # Extract the group ID from the audio path
    # data/UserStudy1-AN-11182019/... => UserStudy1
    tab['group'] = tab['audio_path'].apply(
        lambda x: int(x.split('/')[1].split('-')[0].replace("UserStudy", "")))

    # Identify the task order (AN or NA)
    gp_task_orders = {
        1: 'A-N', 2: 'A-N', 3: 'N-A',
        4: 'N-A', 5: 'N-A', 6: 'A-N',
        7: 'N-A', 8: 'A-N', 9: 'N-A',
        10: 'A-N', 11: 'N-A', 12: 'A-N',
        13: 'N-A',
    }
    tab['task_order'] = tab['group'].apply(
        lambda x: gp_task_orders.get(x, "Unknown")
    )

    # Misc features

    tab['user'] = tab['short_id'].apply(lambda x: x.split('-')[0])

    tab['txt'] = None
    if trans_type == 'machine':
        tab['txt'] = tab['txt_content']
    elif trans_type == 'human':
        tab['txt'] = tab['txt_r_content']

    # Text transcription features

    # Add length of text
    tab['txt_len'] = tab['txt'].str.strip().str.len()
    tab['is_txt_empty'] = (tab['txt_len'] == 0)

    # Add number of words seperated by whitespace
    tab['num_spaced_words'] = None
    if trans_type == 'machine':
        tab['num_spaced_words'] = (tab['txt'].str.count(' ') + 1)
    elif trans_type == 'human':
        tab['num_spaced_words'] = (tab['txt'].str.count(' ') - tab['txt'].str.count(':'))
    tab.loc[tab['is_txt_empty'], 'num_spaced_words'] = 0

    # Add audio duration
    tab['audio_len'] = tab['audio_path'].apply(get_wav_duration)
    tab['num_spwords_psec'] = tab['num_spaced_words'] / tab['audio_len']

    # Add number of portions in transcription
    tab['num_portions'] = None
    if trans_type == 'machine':
        tab['num_portions'] = tab['txt'].str.count('\|') + 1
    elif trans_type == 'human':
        tab['num_portions'] = tab['txt'].str.count(':')
    tab.loc[tab['is_txt_empty'], 'num_portions'] = 0

    tab['timestamp'] = pd.to_datetime(
        tab['audio_st_time'], format='%m%d%Y%H%M%S')

    # "task" means in which task the audio clip was recorded. It can be one of the
    # following:
    #   1A (task1, AR), 2N (task2, NonAR),
    #   1N (task1, NonAR), 2A (task2, AR),
    #   X (a transition between tasks), or Unknown.
    # tab['task'] = 'Unknown'
    tab['task'] = tab['audio_path'].apply(lambda x: taskmap.get(x, 'Unknown'))

    # Sort the table based on the timestamp and group number
    tab.sort_values(['group', 'short_id', 'timestamp', ], inplace=True)

    # Output columns
    output_cols = [
        # Study details
        'group', 'group_id', 'task_order', 'task',
        'user_id', 'short_id', 'user', 'role',
        # Transcription
        'timestamp', 'txt_path', 'txt',
        # Text features
        'num_spaced_words', 'num_spwords_psec', 'num_portions',
        # Audio
        'audio_path', 'audio_len',
        # Task log data
        'log_path',
    ]
    return tab[output_cols].copy()


if __name__ == '__main__':

    # Set parameters
    output_fname = 'data.v0.csv'
    trans_type = 'human'
    only_keep_master_records = True

    # Audio info and transcriptions
    print("[INFO] Getting audio info and transcriptions")
    input_df = gen_audio_info_tab()
    output_df = input_df.copy()
    # Append player's roles
    print("[INFO] Assigning player roles")
    output_df = append_player_roles(output_df)
    # Append some meta data about audio files
    print("[INFO] Extracting and appending meta data")
    output_df = append_meta_data(output_df, trans_type=trans_type)
    # Filter
    if only_keep_master_records:
        output_df = output_df.query('role == "P1"').copy()
    # Output
    output_df.to_csv(output_fname, index=None)
    print("Saved as {}".format(output_fname))

