#!/usr/bin/env python3

from argparse import ArgumentParser

# from precise.util import activate_notify
from precise_runner import PreciseRunner, PreciseEngine
from threading import Event
from itertools import zip_longest
import pyaudio
import wave
import webrtcvad

# from future import print_function
# from pykaldi_test import KaldiInfer
import numpy as np
from matplotlib import pyplot as plt
import pylab
import time
from datetime import datetime

# Deepspeech imports
import deepspeech

# kaldi imports
from kaldi.asr import NnetLatticeFasterOnlineRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
from kaldi.nnet3 import NnetSimpleLoopedComputationOptions
from kaldi.online2 import (
    OnlineEndpointConfig,
    OnlineIvectorExtractorAdaptationState,
    OnlineNnetFeaturePipelineConfig,
    OnlineNnetFeaturePipelineInfo,
    OnlineNnetFeaturePipeline,
    OnlineSilenceWeighting,
)
from kaldi.util.options import ParseOptions
from kaldi.util.table import SequentialWaveReader
from kaldi.matrix import SubVector, Vector


class SpeechRecon(object):
    def __init__(self, VAD_buffer=5, record=False):
        """ Complete End to end Speech recognition with pykaldi(speech recognition), Webrtcvad(Voice activity Detector), and wakeword detector(Mycroft Precise)
        Use Case:
        >>> speech_pipeline = SpeechRecon(record=False)
        >>> speech_pipeline.run()

        Record option True will record audio
        VAD_buffer handles chunk activity recognition sensitivity (currently __ seconds)
        TODO
        VAD_sensitivity handles how many True values vad recognises (currently __ seconds)
        """

        # self.kaldi = KaldiInfer()
        # Record function to init recording currently saved in recordings/
        self.record = record

        # Kaldi paths (currently hardcoded, have to be reinitialized)
        model_path = (
            "/home/prajwaljpj/projects/kaldi/egs/aspire/s5/exp/chain/tdnn_7b/final.mdl"
        )
        graph_path = "/home/prajwaljpj/projects/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/HCLG.fst"
        symbol_path = "/home/prajwaljpj/projects/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/words.txt"
        online_conf = "/home/prajwaljpj/projects/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/conf/online.conf"

        # Precise Wake word detection paths (add location of the precise engine and location of your model)
        precise_engine = PreciseEngine(
            "/home/prajwaljpj/projects/sophia/precise/precise-engine/precise-engine",
            "/home/prajwaljpj/projects/sophia/PyKaldi-EndtoEnd-Recognition/models/hey-computer.net",
        )

        self.sample_rate = 8000
        self.chunk_size = 2048
        self.pa = pyaudio.PyAudio()
        self.precise_pa = pyaudio.PyAudio()
        # print(self.pa.get_sample_size(pyaudio.paInt16))
        # For Kaldi the sampling rate should be 8kHz
        self.stream = self.pa.open(
            self.sample_rate,
            1,
            pyaudio.paInt16,
            True,
            frames_per_buffer=self.chunk_size,
        )
        # For precise The sampling rate should be 16kHz
        self.precise_stream = self.precise_pa.open(
            16000, 1, pyaudio.paInt16, True, frames_per_buffer=self.chunk_size
        )
        self.wakeword_detector = PreciseRunner(
            precise_engine,
            sensitivity=0.5,
            stream=self.precise_stream,
            on_activation=self.precise_activation,
            trigger_level=2,
        )
        self.vad_frame_duration = 10
        self.vad = webrtcvad.Vad(2)
        self.current_activity_states = [False] * VAD_buffer
        self.current_target_state = False  # statement and not command to sophia

        # Deepspeech recogniser Init Options if you want to use deepspeech
        # TODO Fix bugs
        # self.ds_model = deepspeech.Model(
        #     "/home/rbccps2080ti/projects/speech/deepspeech/deepspeech-0.7.0-models.pbmm"
        # )
        # self.ds_model.setBeamWidth(500)
        # self.ds_model.enableDecoderWithLM("/home/rbccps2080ti/projects/speech/deepspeech/deepspeech-0.6.1-models/lm.binary",
        # "/home/rbccps2080ti/projects/speech/deepspeech/deepspeech-0.6.1-models/trie",
        # 0.75,
        # 1.85)

        # Kaldi Init options
        feat_opts = OnlineNnetFeaturePipelineConfig()
        endpoint_opts = OnlineEndpointConfig()
        po = ParseOptions("")
        feat_opts.register(po)
        endpoint_opts.register(po)
        po.read_config_file(online_conf)
        self.feat_info = OnlineNnetFeaturePipelineInfo.from_config(feat_opts)

        # Construct recognizer
        decoder_opts = LatticeFasterDecoderOptions()
        decoder_opts.beam = 13
        decoder_opts.max_active = 7000
        decodable_opts = NnetSimpleLoopedComputationOptions()
        decodable_opts.acoustic_scale = 1.0
        decodable_opts.frame_subsampling_factor = 3
        decodable_opts.frames_per_chunk = 150
        self.asr = NnetLatticeFasterOnlineRecognizer.from_files(
            model_path,
            graph_path,
            symbol_path,
            decoder_opts=decoder_opts,
            decodable_opts=decodable_opts,
            endpoint_opts=endpoint_opts,
        )

    def listen(self):
        """ This method calls stream.read to read a chunk of data.
        Chunk size can be set in the init method manually"""
        return self.stream.read(self.chunk_size)

    def VAD(self, stream_data, frame_duration=10):
        """ Voice activity detector for that chunk.
        Stream read records data in bytearray.
        The data for the VAD must be int16 array (or np.Int16 array)"""

        # Array of size 2048 has to be divided into pieces of 160 for 8kHz frequency
        mini_chunks_size = (
            int(self.sample_rate * frame_duration / 1000) * 2
        )  # 320 and for 8k it is 160

        def chunker(chunk, n=mini_chunks_size, fillvalue=0):
            # Sub method to divide 2048 to 320/160 length array for the VAD to process
            args = [iter(chunk)] * n
            return zip_longest(*args, fillvalue=fillvalue)

        activity = []

        for mini_chunk in chunker(stream_data):
            mchk = bytes(mini_chunk)
            activity.append(self.vad.is_speech(mchk, self.sample_rate))

        self.update_activity(any(activity))

        return any(activity)

    def update_activity(self, activity):
        """ Method to keep track of whether there is voice activity or not through a class variable """
        assert type(activity) == bool
        self.current_activity_states.append(activity)
        self.current_activity_states = self.current_activity_states[1:]

    def update_target(self, state):
        """ Method to kep track of whether the sentence is a command or a statement.
        In the sense if the uttered speech is a command to the robot or is a statement to be said through the robot """
        assert type(state) == bool
        self.current_target_state = state

    def precise_activation(self):
        """ Extra method as precise activation is a lambda function """
        print("Precise Activation Occured")
        self.update_target(True)

    def init_decoder(self, kaldidecoder=True):
        """ Initialize some parementers of the decoder.
        For Kaldi, creating an input stream pipeline and initializing the decoder object,
        For deepspeech, creating a Stream object through which the chunks are decoded"""
        if kaldidecoder:
            self.feature_pipeline = OnlineNnetFeaturePipeline(self.feat_info)
            self.asr.set_input_pipeline(self.feature_pipeline)
            self.asr.init_decoding()
        else:
            self.context = self.ds_model.createStream()
            # print("deepspeech not yet implemented")

    def decode_chunk(self, chunk, kaldidecoder=True, last_chunk=False):
        """ Method to decode audio chunks one by one.
        For kaldi, it is necessary to know of the chunk is last one or not.
        This is provided by the VAD and the current_activity_states class variable.
        For deepspeech, last chunk is not required. But Stream end is required which is handled by the destroy decoder """
        if kaldidecoder:
            chunk = SubVector(chunk)
            if chunk.is_zero():
                print("data is zero")
            self.feature_pipeline.accept_waveform(8000.0, chunk)
            if last_chunk:
                self.feature_pipeline.input_finished()
            self.asr.advance_decoding()
            if not last_chunk:
                out = self.asr.get_partial_output()
                return out["text"]
            return ""
        else:
            self.ds_model.feedAudioContent(self.context, chunk)
            out = self.ds_model.intermediateDecode(self.context)
            return out

    def destroy_decoder(self, kaldidecoder=True):
        """ Method to complete decoding as for that utterance."""
        if kaldidecoder:
            # self.feature_pipeline.input_finished()
            self.asr.finalize_decoding()
            out = self.asr.get_output()
            return out["text"]
        else:
            out = self.ds_model.finishStream(self.context)
            return out

    def run(self):
        """ Run Method to use to run the whole pipeline indefinately
        __Use this method for your project__"""
        # initialize precise
        self.wakeword_detector.start()

        while True:
            # self.wakeword_detector.start()
            self.wakeword_detector.play()
            chunk = self.listen()
            if self.record:
                # If you want to record audio snippets of utterances
                # print("LOG :: Recording audio clips in $PWD/recordings/")
                total_data_stream = []
                datetimeObj = datetime.now()
                filename = (
                    "recordings/"
                    + str(datetimeObj.day)
                    + "-"
                    + str(datetimeObj.month)
                    + "-"
                    + str(datetimeObj.year)
                    + "_"
                    + str(datetimeObj.hour)
                    + "-"
                    + str(datetimeObj.minute)
                    + "-"
                    + str(datetimeObj.second)
                    + ".wav"
                )
                wf = wave.open(filename, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                total_data_stream.append(chunk)

            self.init_decoder(kaldidecoder=True)
            chunk_array = np.asarray(np.fromstring(chunk, dtype=np.int16))
            activity = self.VAD(chunk)
            self.update_activity(activity=activity)
            # print(self.current_activity_states)
            # print("outer While")
            # print("Speech Activity: ", any(self.current_activity_states))

            # secondary loop to check if there is audio activity
            while (
                self.current_activity_states.count(True) > 2
            ):  # Tune this as per your requirements
                # print("inner while loop")
                chunk = self.listen()
                if self.record:
                    total_data_stream.append(chunk)
                chunk_array = np.asarray(np.fromstring(chunk, dtype=np.int16))
                activity = self.VAD(chunk)
                # print(self.current_activity_states)
                if self.current_activity_states[0] and not any(
                    self.current_activity_states[3:]
                ):  # Tune this as per your requirements
                    # Last frame to decode
                    self.update_activity(activity=activity)
                    self.decode_chunk(chunk_array, kaldidecoder=True, last_chunk=True)
                    final_output = self.destroy_decoder(kaldidecoder=True)
                    ## TODO Integrate precise
                    # self.wakeword_detector.pause()
                    print(self.current_target_state)
                    if self.current_target_state:
                        print("Command: ", final_output)
                        self.update_target(False)
                    else:
                        print("Statement: ", final_output)
                    # print("Output : " + final_output)
                    if self.record:
                        wf.writeframes(b"".join(total_data_stream))
                        wf.close()
                        # print("LOG:: Recorded at: ", filename)
                    break
                partial_output = self.decode_chunk(
                    chunk_array, kaldidecoder=True, last_chunk=False
                )
                # print("Partial output: " + partial_output)


if __name__ == "__main__":
    speech_pipeline = SpeechRecon(record=False)
    speech_pipeline.run()
