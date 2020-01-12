#!python2.7

from pathlib import Path
import pandas as pd
import base64
from google.cloud import speech_v1
from google.cloud import speech_v1p1beta1
from google.cloud.speech_v1 import enums as sp_enums
from google.cloud import language_v1
from google.cloud.language_v1 import enums as lang_enums
import io

from utils import gen_audio_info_tab


class AudioRecognizer:
    """
    An audio recognizer based on GCloud service. Before using this class, you
    have to set the environment variable GOOGLE_APPLICATION_CREDENTIALS with
    the path to your credential key downloaded from the GCloud dashboard.
    """

    CONFIG_CONTEXT = 1
    CONFIG_NO_CONTEXT = 2
    CONFIG_TIME = 3

    def __init__(self):
        """
        Init GCloud SpeechClient and the configuration.
        """
        self.client = speech_v1.SpeechClient()
        self.beta_client = speech_v1p1beta1.SpeechClient()

        language_code = "en-US"
        sample_rate_hertz = 16000
        encoding = sp_enums.RecognitionConfig.AudioEncoding.LINEAR16
        use_enhanced = True
        model = "phone_call"

        diarization_speaker_count = 2
        enable_speaker_diarization = True

        enable_word_time_offsets = True

        speech_contexts = [{
            "phrases": [
                "climb up", "climb down", "move forward", "move backward",
                "up", "down", "forward", "backward",
                "climbing off", "three",
                "hover", "engine", "start", "stop", "what", "why", "which",
                "how", "where", "block", "distance", "continue", "second",
                "go up", "go down", "wait", "make it", "should be",
                "this is", "running", "try again", "run", "reset", "high",
                "wall", "capture the flags", "flag", "capture",
                "I think", "I got", "we don't need", "continue for",
            ]
        }]

        self.config = {
            "model": model,
            "use_enhanced": use_enhanced,
            "language_code": language_code,
            "sample_rate_hertz": sample_rate_hertz,
            "encoding": encoding,
            "speech_contexts": speech_contexts,
        }

        self.no_context_config = {
            "model": model,
            "use_enhanced": use_enhanced,
            "language_code": language_code,
            "sample_rate_hertz": sample_rate_hertz,
            "encoding": encoding,
        }

        self.diaz_config = {
            "model": model,
            "use_enhanced": use_enhanced,
            "language_code": language_code,
            "sample_rate_hertz": sample_rate_hertz,
            "encoding": encoding,
            "speech_contexts": speech_contexts,
            "diarization_speaker_count": diarization_speaker_count,
            "enable_speaker_diarization": enable_speaker_diarization,
        }

        self.time_config = {
            "enable_word_time_offsets": enable_word_time_offsets,
            "model": model,
            "use_enhanced": use_enhanced,
            "language_code": language_code,
            "sample_rate_hertz": sample_rate_hertz,
            "encoding": encoding,
            "speech_contexts": speech_contexts,
        }

    def _recognize_long(self, audio):
        """
        Recognize a long audio clip (>1min)
        """
        return self.client.long_running_recognize(self.config, audio)

    def _recognize_short(self, audio, context=True):
        """
        Recognize a short audio clip (<=1min)
        """
        if context:
            return self.client.recognize(self.config, audio)
        else:
            return self.client.recognize(self.no_context_config, audio)

    def _recognize_diarize_short(self, audio):
        """
        Recognize and diarize a short audio clip (<=1min)

        Reference:
            https://cloud.google.com/speech-to-text/docs/multiple-voices
        """
        return self.beta_client.recognize(self.diaz_config, audio)

    def encode_audio_b64(self, fname):
        """
        Convert an audio file to base64 encoding.
        """
        with open(str(fname), 'rb') as fin:
            rt = base64.b64encode(fin.read())
        return rt

    def recognize_long_audio(self, filename):
        """
        Transcribe a long audio file (>1min).
        Reference: https://cloud.google.com/speech-to-text/docs/async-recognize
        """
        audio = {"content": self.encode_audio_b64(filename)}
        operation = self._recognize_long(audio)
        print(u"Waiting for operation to complete...: {}".format(filename))

        response = operation.result()
        trscript = []
        for result in response.results:
            alternative = result.alternatives[0]
            print(alternative)
            print(u"Transcript: {}".format(alternative.transcript))
            trscript.append(alternative.transcript)

        return trscript

    def recognize_short_audio(self, filename, diarization=False, context=True):
        """
        Transcribe a short audio file (<1min).

        Reference: https://cloud.google.com/speech-to-text/docs/sync-recognize
        """
        with io.open(filename, "rb") as f:
            content = f.read()
        audio = {"content": content}

        trscript = []
        if diarization:
            response = self._recognize_diarize_short(audio)
        else:
            response = self._recognize_short(audio, context=context)

        for result in response.results:
            alternative = result.alternatives[0]
            trscript.append(alternative.transcript)

        return trscript


def transcribe_audio_files(overwrite=False, verbose=False):
    """
    Transcribe all audio files in the data folder.

    Args:

        overwrite: Whether to overwrite the existing transcript if exists.
        verbose: Whether to print out the transcript when processing files.

    """
    audio_recognizer = AudioRecognizer()
    rt = gen_audio_info_tab()

    cnt_total = rt.shape[0]
    cnt_progress = 0
    for i, row in rt.iterrows():
        cnt_progress += 1
        print(u"Progress ({}/{})".format(cnt_progress, cnt_total))

        waiting_list = [
            (
                AudioRecognizer.CONFIG_CONTEXT, row['audio_path'],
                row['txt_path'], row['txt_content'],
            ),
            # (
            #     AudioRecognizer.CONFIG_NO_CONTEXT, row['audio_path'],
            #     row['txt_dft_path'], row['txt_dft_content'],
            # ),
            # (
            #     AudioRecognizer.CONFIG_TIME, row['audio_path'],
            #     row['txt_time_path'], row['txt_time_content'],
            # ),
        ]

        for context, filename, txt_path, txt_content in waiting_list:

            # If the audio hasn't been transcribed
            if overwrite or txt_content == None:
                print(u"Processing: {} {}".format(context, filename))

                if context == AudioRecognizer.CONFIG_NO_CONTEXT:
                    txt = audio_recognizer.recognize_short_audio(
                        filename, context=False)
                else:
                    txt = audio_recognizer.recognize_short_audio(
                        filename, context=True)

                if verbose:
                    print(u"{}".format(txt))

                with open(txt_path, "w+") as fout:
                    fout.write("|".join(txt))
                    fout.flush()

            else:
                if verbose:
                    print(u"Skipping...: {}".format(filename))
                    print(u"{}".format(txt_content))


if __name__ == '__main__':
    # transcribe_audio_files(overwrite=False, verbose=True)
    # transcribe_audio_files()
    pass
