#!python2.7

from pathlib import Path
import pandas as pd
import base64
from google.cloud import language_v1
from google.cloud.language_v1 import enums as lang_enums
import io
import ast
import time


class LanguageAnalyzer:
    """
    A language analyzer based on GCloud Natural Language API. Now it is used to analyze sentiment only.
    """

    def __init__(self):
        self.client = language_v1.LanguageServiceClient()
        self.type_ = lang_enums.Document.Type.PLAIN_TEXT
        self.language = "en"
        self.encoding_type = lang_enums.EncodingType.UTF8

    def pack_content(self, txt_content):
        """
        Pack and transform the txt_content into the input format of GCloud requests.
        """
        document = {
            "content": txt_content,
            "type": self.type_,
            "language": self.language,
        }
        return document

    def analyze_sentiment(self, txt_content, verbose=False):
        """
        Analyze the sentiment in txt_content by GCloud API.

        Reference:
            https://cloud.google.com/natural-language/docs/analyzing-sentiment
        """
        doc = self.pack_content(txt_content)
        response = self.client.analyze_sentiment(
            doc, encoding_type=self.encoding_type)

        if verbose:
            print(u"Document sentiment score: {}".format(
                response.document_sentiment.score))
            print(u"Document sentiment magnitude: {}".format(
                response.document_sentiment.magnitude))

            for sentence in response.sentences:
                print(u"Sentence text: {}".format(sentence.text.content))
                print(u"Sentence sentiment score: {}".format(
                    sentence.sentiment.score))
                print(u"Sentence sentiment magnitude: {}".format(
                    sentence.sentiment.magnitude))

        return response

    def extract_result_senti(self, rt):
        doc_senti = rt.document_sentiment
        sent_senti = []

        for s in rt.sentences:
            sent_senti.append(
                (s.sentiment.magnitude, s.sentiment.score, s.text.content, ))

        return pd.Series({
            'senti_doc_magnitude': doc_senti.magnitude,
            'senti_doc_score': doc_senti.score,
            'senti_sent': sent_senti,
        })


if __name__ == '__main__':
    lang_analyzer = LanguageAnalyzer()

    data = pd.read_csv('data.v1.csv')
    data['sent'] = data['sent'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    data_gp = data.groupby(['task_order', 'task', 'user'])[
        'sent'].apply(lambda x: '. '.join([i for i in x])).reset_index()
    sent_docs = data_gp['sent']

    print(u"Start parsing sentiment...")
    cnt_request = 0
    senti_docs = []
    for x in sent_docs:
        print("Progress {} / {}".format(cnt_request, len(sent_docs)))
        cnt_request += 1
        rt = lang_analyzer.analyze_sentiment(x)
        senti_docs.append(rt)

        if cnt_request % 300 == 0:
            print(u"Sleep 1 min to keep the quota...")
            time.sleep(60)

    # Process the result

    senti_records = [lang_analyzer.extract_result_senti(i) for i in senti_docs]
    senti_df = pd.DataFrame.from_records(senti_records)
    senti_df['senti_sent_magnitude'] = senti_df['senti_sent'].apply(
        lambda x: pd.Series([i[0] for i in x]).mean())
    senti_df['senti_sent_score'] = senti_df['senti_sent'].apply(
        lambda x: pd.Series([i[1] for i in x]).mean())

    senti_df.to_csv('senti_df.user.csv')

    # Merge and export

    def standarize(data_series):
        return (data_series - data_series.mean()) / data_series.std()

    datav2_user = data_gp.copy()
    audiolen_df = data.groupby(['task_order', 'task', 'user'])[
        'audio_len'].apply(sum).reset_index()

    datav2_user = pd.merge(datav2_user[['task_order', 'task', 'user']], senti_df,
                           left_index=True, right_index=True)
    datav2_user = pd.merge(datav2_user, audiolen_df['audio_len'],
                           left_index=True, right_index=True)

    datav2_user['senti_sent_score'] /= datav2_user['audio_len']
    datav2_user['senti_sent_magnitude'] /= datav2_user['audio_len']
    datav2_user['senti_sent_product'] = datav2_user['senti_sent_score'] * \
        datav2_user['senti_sent_magnitude']
    datav2_user['senti_sent_product'] = standarize(
        datav2_user['senti_sent_product'])

    datav2_user.to_csv('data.v2.user.csv', index=None)

    # senti_df = pd.read_csv('senti_df.csv', index_col=0)
    # senti_df['senti_sent'] = senti_df['senti_sent'].apply(ast.literal_eval)
    # senti_list = []
    # for i in senti_df.senti_sent:
    #     for j in i:
    #         senti_list.append(j)
    # senti_list = pd.DataFrame.from_records(
    #     senti_list, columns=('score', 'magnitude', 'txt', ))
    # senti_list.to_csv('senti_sent.list.csv', index=None)

    # datav2 = data.copy()
    # datav2 = pd.merge(datav2, senti_df, left_index=True, right_index=True)
    # datav2['senti_sent_score'] /= datav2['audio_len']
    # datav2['senti_sent_magnitude'] /= datav2['audio_len']
    # datav2.to_csv('data.v2.csv', index=None)
