#!python3

"""
This script parses the audio info table with metadata (generate by the script "preprocess_trsrpt_meta.py") and extracts several NLP features.
"""

import os
import pandas as pd
from pathlib import Path
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
from nltk.parse import CoreNLPParser
from nltk.parse.corenlp import CoreNLPServer


class CoreNLPSentenceAnalyzer():
    """
    A sentence analyzer based on Stanford CoreNLP.

    Refernces:
        The CoreNLP Syntax Parser
            https://bbengfort.github.io/snippets/2018/06/22/corenlp-nltk-parses.html
        Penn Treebank II Tags
            https://gist.github.com/nlothian/9240750
    """

    def __init__(self):
        self.lab_set = set()

    def init_server(self):
        STANDFORD = os.path.join("stanford-corenlp-full-2018-10-05")
        self.server = CoreNLPServer(
            os.path.join(STANDFORD, "stanford-corenlp-3.9.2.jar"),
            os.path.join(STANDFORD, "stanford-corenlp-3.9.2-models.jar")
        )
        self.server.start()
        self.parser = CoreNLPParser()

    def stop_server(self):
        self.server.stop()

    def parse_syntax(self, sent):
        return next(self.parser.raw_parse(sent))

    def _collect_labels(self, node):
        """
        Collect labels in the given node recursively. This method should not be invoked directly but done by collect_labels.
        """
        try:
            self.lab_result.append(node.label())
        except AttributeError:
            return
        for nn in node:
            self._collect_labels(nn)
        return

    def collect_labels(self, node):
        """
        Collect all labels in a tree starting from the given node.
        """
        self.lab_result = []  # used to collect labels in the recursion
        self._collect_labels(node)
        lab_counter = Counter(self.lab_result)

        # Keep the tags we have seen so far
        self.lab_set = self.lab_set.union(lab_counter.keys())

        return lab_counter

    def get_lab_series(self, lab_counter_list):
        """
        Convert and merge all lab_counters in the given list (the result of "collect_labels") into a series by using tags which have been seen so far (self.lab_set).
        """
        rt = pd.DataFrame(columns=self.lab_set)
        for lab_counter in lab_counter_list:
            rt = rt.append(pd.Series(lab_counter, index=self.lab_set),
                           ignore_index=True)
        rt = rt.add_prefix('penn_')
        return rt.sum()


def preprocess_txt_time(df):
    """
    Preprocess txt and timestamps in df.
    """
    def _replace_words(x, r_list):
        """
        (For apply) Replace words in x by following rules in r_list
        """
        for r in r_list:
            x = x.replace(r[0], r[1])
        return x

    def _split_txt_by_speaker(x):
        """
        (For apply) Split a txt cell into a two-item tuple indexed by speakers.
        """
        txt_parts = x.split('\n')
        s1_txt, s2_txt = [], []
        for part in txt_parts:
            _part = part.strip()
            if len(_part) > 0:
                if _part[0] == '1':
                    s1_txt.append(_part.replace('1:', '').strip())
                elif _part[0] == '2':
                    s2_txt.append(_part.replace('2:', '').strip())
        s1_txt, s2_txt = ' '.join(s1_txt), ' '.join(s2_txt)
        return (s1_txt, s2_txt)

    def _explode_speaker_txt_into_rows(x):
        """
        (For apply) Split a row into two rows indexed by speakers.
        """
        rt = pd.DataFrame([x]).explode(
            'speaker_txt_tuple').reset_index(drop=True)
        rt.loc[0, 'speaker'] = 'S1'
        rt.loc[1, 'speaker'] = 'S2'
        return rt

    print("[INFO] Preprocessing text and timestamps")

    # Copy the input data
    data = df.copy()

    # Filter
    data = data.loc[data['task'] != 'X']
    data['txt'] = data['txt'].fillna('')

    # Group the content of "txt" by speakers
    # (i.e., explode one transcription into multiple rows)
    print('[INFO] Exploding txt into one speaker per row')
    data['speaker'] = "None"
    data['speaker_txt_tuple'] = data['txt'].apply(_split_txt_by_speaker)
    data1 = []
    for _, row in data.iterrows():
        data1.append(_explode_speaker_txt_into_rows(row))
    data = pd.concat(data1).reset_index(drop=True)
    # Remove intermediate redundant columns
    data['txt'] = data['speaker_txt_tuple']
    data.drop(['speaker_txt_tuple'], axis=1, inplace=True)

    print('[INFO] Processing txt')

    # Backup the original txt
    data['raw_txt'] = data['txt']

    # Replace some abbreviations
    replace_word_list = [
        ("n't", " not"), ("'ll", " will"), ("'ve", " have"), ("'d", " would"),
        ("'m", " am"), ("'s", " is"), ("'re", " are"), ("\"", " "), ("'", " "),
        ('cuz', 'because'), ("Ford", "forward"), ('gonna', 'going to'),
        ('wanna', 'want to'), ('...', '.'), ('..', '.'), ('sec', 'second'),
        ('umhmm', '<confirm>'), ('yeah', '<confirm>'),
    ]
    data['txt'] = data['txt'].apply(
        lambda x: _replace_words(x, replace_word_list))
    data['txt'] = data['txt'].str.lower()
    # Remove speaker annotation
    data['txt'] = data['txt'].apply(
        lambda x: x.replace('1:', '').replace('2:', ''))

    # Tokenized text
    tokenizer = RegexpTokenizer(r'\w+')
    data['tk'] = data['txt'].apply(
        lambda x: tokenizer.tokenize(x.replace("|", "")))
    # Tokens without stopwords
    en_stopws = stopwords.words('english')
    data['tk_nostop'] = data['tk'].apply(
        lambda x: [i for i in x if i not in en_stopws])

    # Stemmed tokens
    wnl = WordNetLemmatizer()
    data['tk_wnet'] = data['tk_nostop'].apply(
        lambda x: [wnl.lemmatize(i) for i in x])
    stemmer = SnowballStemmer("english")
    data['tk_snbl'] = data['tk_nostop'].apply(
        lambda x: [stemmer.stem(i) for i in x])

    # Misc features of keywords
    data['num_tk'] = data['tk'].apply(lambda x: len(x))
    data['num_tknstop'] = data['tk_nostop'].apply(lambda x: len(x))
    data['num_tkwnet'] = data['tk_wnet'].apply(lambda x: len(x))
    data['num_tksnbl'] = data['tk_snbl'].apply(lambda x: len(x))
    data['rate_tk'] = data['num_tk'] / data['audio_len']
    data['rate_tknstop'] = data['num_tk'] / data['audio_len']

    print('[INFO] Processing timestamps')

    # Recover the datatype of timestamp
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # Normalize timestamps by user
    user_start_time = data.groupby('user_id')['timestamp'].min()
    data['time_offset'] = data.groupby('user_id')['timestamp'].transform(
        lambda x: (x - x.shift(1))).dt.total_seconds().fillna(0).astype("int")
    data['time_cumsum'] = data.groupby('user_id')['time_offset'].transform(
        lambda x: x.cumsum())

    tk_columns = ('tk', 'tk_nostop', 'tk_wnet', 'tk_snbl',)

    return data, tk_columns


def add_keyword_features(df, input_column='tk_wnet',
                         f_kw_cmd='keywords_cmds',
                         f_kw_action='keywords_actions'):
    """
    Annotate and count keywords.
    """
    print("[INFO] Adding keyword features by files: {}, {}".format(
        f_kw_cmd, f_kw_action))

    # Copy the input data
    data = df.copy()
    if input_column not in data.columns:
        raise IndexError(
            "The input column was not found: {}".format(input_column))

    # Load the keyword files
    print('[INFO] Loading the predefined keyword files')
    if not (Path(f_kw_cmd).exists() and Path(f_kw_action).exists()):
        raise FileNotFoundError("The specified keyword files were not found.")

    # Count command keywords
    with open(f_kw_cmd, 'r') as fin:
        keywords_cmds = [i.strip() for i in fin.readlines()]
    data['num_kw_cmds'] = data[input_column].apply(
        lambda x: sum([v for k, v in Counter(x).items() if k.lower() in keywords_cmds]))

    # Count action keywords
    with open(f_kw_action, 'r') as fin:
        keywords_actions = [i.strip() for i in fin.readlines()]
    data['num_kw_actions'] = data[input_column].apply(
        lambda x: sum([v for k, v in Counter(x).items() if k.lower() in keywords_actions]))

    # Normalize the numbers
    data['rate_kw_cmds'] = data['num_kw_cmds'] / data['audio_len']
    data['rate_kw_actions'] = data['num_kw_actions'] / data['audio_len']

    return data


def add_pos_features(df, input_column='tk_nostop'):
    """
    Add NLTK general PoS tags.
    """
    def _cnt_nltk_pos(x):
        """
        Count numbers of pos tags in the tuple list x. x should look like:
        [
            ('soon', 'ADV'), ('I', 'PRON'), ('move', 'VERB'),
            ('forward', 'ADV'), ('grab', 'ADJ'), ('flag', 'NOUN'),
            ('Maybe', 'ADV')
        ].
        """
        rt = {
            "ADJ": 0, "ADP": 0, "ADV": 0, "CONJ": 0, "DET": 0,
            "NOUN": 0, "NUM": 0, "PRT": 0, "PRON": 0, "VERB": 0,
            ".": 0, "X": 0,
        }
        for k, p in x:
            rt[p] += 1
        return pd.Series(rt)

    print("[INFO] Adding NLTK PoS tags")

    # Copy the input data
    data = df.copy()
    if input_column not in data.columns:
        raise IndexError(
            "The input column was not found: {}".format(input_column))

    # Add PoS tags
    # Reference: https://www.nltk.org/book/ch05.html
    data['tk_pos'] = data[input_column].apply(
        lambda x: pos_tag(x, tagset='universal'))

    # Count PoS tags
    pos_tag_cols = ["ADJ", "ADP", "ADV", "CONJ", "DET",
                    "NOUN", "NUM", "PRT", "PRON", "VERB", ".", "X", ]
    pos_nums = data['tk_pos'].apply(_cnt_nltk_pos)

    # Append the pos columns by indices
    data = pd.merge(data, pos_nums, left_index=True, right_index=True)
    for i in pos_tag_cols:
        data[i] = data[i] / data['audio_len']

    return data


def add_corenlp_pos_features(df, input_column='txt'):
    """
    Add Treebank PoS tags from CoreNLP.
    """
    # Copy the input data
    data = df.copy()
    if input_column not in data.columns:
        raise IndexError(
            "The input column was not found: {}".format(input_column))

    print("[INFO] Initalizing and starting the CoreNLP server")
    core_nlp_analyzer = CoreNLPSentenceAnalyzer()
    core_nlp_analyzer.init_server()

    # Identify sentences
    print("[INFO] Identifying sentences")
    data['sent'] = data[input_column].apply(lambda x: sent_tokenize(
        x.replace('1:', '').replace('2:', '').replace('\n', '.')
        .replace('..', '.').replace(',', '.')))
    # Parse sentence trees and annotate PoS tags
    print("[INFO] Parsing sentence trees")
    data['sent_tree'] = data['sent'].apply(
        lambda x: [core_nlp_analyzer.parse_syntax(i) for i in x])
    # Collect PoS tags in the sentence trees recursively
    print("[INFO] Collecting PoS tags in the sentence trees")
    tmp = data['sent_tree'].apply(
        lambda x: [core_nlp_analyzer.collect_labels(i) for i in x])
    # Convert the counter dict into pd.Series
    sent_tree_lab_cnts = tmp.apply(core_nlp_analyzer.get_lab_series)

    # Merge all columns of PoS counts back to the dataframe
    data = pd.merge(data, sent_tree_lab_cnts,
                    left_index=True, right_index=True)
    # Normalize the numbers
    for i in sent_tree_lab_cnts.columns:
        data[i] = data[i] / data['audio_len']

    # Don't forget to stop the CoreNLP server
    # (if it does not stop, you need to do it on your own)
    core_nlp_analyzer.stop_server()

    return data


if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data.v0.csv')

    # Preprocess txt and timestamps
    data, tk_columns = preprocess_txt_time(data)

    # Add keyword features
    data = add_keyword_features(
        data, input_column='tk_wnet',
        f_kw_cmd='keywords_cmds', f_kw_action='keywords_actions')

    # # Add NLTK PoS tags
    # data = add_pos_features(data, input_column='tk_nostop')
    # # Add CoreNLP PoS tags
    # data = add_corenlp_pos_features(data, input_column='txt')

    # Drop redundant columns
    data.drop(['user', 'short_id', 'role', 'txt_path', 'log_path',
               'audio_path'], axis=1, inplace=True)

    # Before exporting the result, convert token lists to strings
    for i in tk_columns:
        data.loc[:, i] = data.loc[:, i].apply(lambda x: ','.join(x))
    # Export the processed dataframe
    output_fname = 'data.v1.csv'
    print("[INFO] The processed data was saved as {}".format(output_fname))
    data.to_csv(output_fname, index=None)
