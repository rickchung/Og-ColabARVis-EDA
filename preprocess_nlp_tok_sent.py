#!python3

"""
This script parses the audio info table with metadata (generate by the script "preprocess_trsrpt_meta.py") and extracts several NLP features.
"""

import os
import pandas as pd
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

    print("[INFO] Preprocessing text and timestamps")

    # Copy the input data
    data = df.copy()

    # Filter
    data = data.loc[data['task'] != 'X']
    data['txt'] = data['txt'].fillna('')

    replace_word_list = [
        ("n't", " not"), ("'ll", " will"), ("'ve", " have"), ("'d", " would"),
        ("'m", " am"), ("'s", " is"), ("'re", " are"), ("\"", " "), ("'", " "),
        ('cuz', 'because'), ("Ford", "forward"), ('gonna', 'going to'),
        ('wanna', 'want to'), ('...', '.'), ('..', '.'), ('sec', 'second'),
        ('umhmm', '<confirm>'), ('yeah', '<confirm>'),
    ]

    # Replace some abbreviations
    data['txt'] = data['txt'].apply(lambda x: _replace_words(x, replace_word_list))
    data['txt'] = data['txt'].str.lower()
    # Backup the original txt with annotated speakers
    data['raw_txt'] = data['txt']
    # Remove speaker annotation
    data['txt'] = data['txt'].apply(lambda x: x.replace('1:', '').replace('2:', ''))

    # Recover the datatype of timestamp
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # Normalize timestamps by user
    data['user'] = data['short_id']
    user_start_time = data.groupby('user')['timestamp'].min()
    data['time_offset'] = data.groupby('user')['timestamp'].transform(
        lambda x: (x - x.shift(1))).dt.total_seconds().fillna(0).astype("int")
    data['time_cumsum'] = data.groupby('user')['time_offset'].transform(
        lambda x: x.cumsum())

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

    return data

def add_keyword_features(df, f_kw_cmd='keywords_cmds', f_kw_action='keywords_actions'):
    """
    Annotate and count keywords.
    """
    data = df.copy()

    print("[INFO] Adding keyword features: {}, {}".format(f_kw_cmd, f_kw_action))

    # Count customized keywords
    with open(f_kw_cmd, 'r') as fin:
        keywords_cmds = [i.strip() for i in fin.readlines()]
    data['num_kw_cmds'] = data['tk_wnet'].apply(
        lambda x: sum([v for k, v in Counter(x).items() if k.lower() in keywords_cmds]))

    with open(f_kw_action, 'r') as fin:
        keywords_actions = [i.strip() for i in fin.readlines()]
    data['num_kw_actions'] = data['tk_wnet'].apply(
        lambda x: sum([v for k, v in Counter(x).items() if k.lower() in keywords_actions]))

    # Normalize
    data['rate_kw_cmds'] = data['num_kw_cmds'] / data['audio_len']
    data['rate_kw_actions'] = data['num_kw_actions'] / data['audio_len']

    return data

def add_pos_features(df, token_column='tk_nostop'):
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

    # Add PoS tags
    # Reference: https://www.nltk.org/book/ch05.html
    data['tk_pos'] = data[token_column].apply(
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

def add_corenlp_pos_features(df):
    """
    Add Treebank PoS tags from CoreNLP.
    """
    data = df.copy()

    print("[INFO] Initalizing and starting the CoreNLP server")
    core_nlp_analyzer = CoreNLPSentenceAnalyzer()
    core_nlp_analyzer.init_server()

    # Identify sentences
    print("[INFO] Identifying sentences")
    data['sent'] = data['txt'].apply(lambda x: sent_tokenize(
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
    data = pd.merge(data, sent_tree_lab_cnts, left_index=True, right_index=True)
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
    data = preprocess_txt_time(data)

    # Add keyword features
    data = add_keyword_features(data)
    # Add NLTK PoS tags
    data = add_pos_features(data)
    # Add CoreNLP PoS tags
    data = add_corenlp_pos_features(data)

    # Export the processed dataframe
    data.to_csv('data.v1.csv', index=None)
