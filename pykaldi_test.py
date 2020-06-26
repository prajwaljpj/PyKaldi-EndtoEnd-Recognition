from __future__ import print_function

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


class KaldiInfer(object):
    def __init__(self, 
        model_path="/home/rbccps2080ti/projects/speech/kaldi/egs/aspire/s5/exp/chain/tdnn_7b/final.mdl", 
        graph_path="/home/rbccps2080ti/projects/speech/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/HCLG.fst", 
        symbol_path="/home/rbccps2080ti/projects/speech/kaldi/egs/aspire/s5/data/lang_chain/words.txt", 
        online_conf="/home/rbccps2080ti/projects/speech/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/conf/online_16k.conf", 
        chunk_size=2048):

        self.chunk_size = chunk_size
        # Add all paths
        # model_path = "/home/rbccps2080ti/projects/speech/kaldi/egs/aspire/s5/exp/chain/tdnn_7b/final.mdl"
        # graph_path = "/home/rbccps2080ti/projects/speech/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/HCLG.fst"
        # symbol_path = "/home/rbccps2080ti/projects/speech/kaldi/egs/aspire/s5/data/lang_chain/words.txt"

        # Define online feature pipeline
        feat_opts = OnlineNnetFeaturePipelineConfig()
        endpoint_opts = OnlineEndpointConfig()
        po = ParseOptions("")
        feat_opts.register(po)
        endpoint_opts.register(po)
        po.read_config_file(
            online_conf
        )
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

    def infer(self, data_path):
        for key, wav in SequentialWaveReader("scp:echo person "+data_path+"|"):
            # print(type(wav))
            # print(wav.samp_freq)
            feat_pipeline = OnlineNnetFeaturePipeline(self.feat_info)
            self.asr.set_input_pipeline(feat_pipeline)
            self.asr.init_decoding()
            data = wav.data()[0]
            # print(data.dim)
            # print(type(data))
            last_chunk = False
            # part = 1
            prev_num_frames_decoded = 0
            for i in range(0, len(data), self.chunk_size):
                if i + self.chunk_size >= len(data):
                    last_chunk = True
                feat_pipeline.accept_waveform(wav.samp_freq, data[i : i + self.chunk_size])
                if last_chunk:
                    feat_pipeline.input_finished()
                self.asr.advance_decoding()
                num_frames_decoded = self.asr.decoder.num_frames_decoded()
                if not last_chunk:
                    if num_frames_decoded > prev_num_frames_decoded:
                        prev_num_frames_decoded = num_frames_decoded
                        out = self.asr.get_partial_output()
                        # print(key + "-part%d" % part, out["text"])
                        print(out["text"])
                        # part += 1
            self.asr.finalize_decoding()
            out = self.asr.get_output()
            # print(key + "-final", out["text"])
            # print(out["text"])
            return out

    def infer_in(self):
        for key, wav in SequentialWaveReader("scp:wav.scp"):
            # print(type(wav))
            # print(wav.samp_freq)
            feat_pipeline = OnlineNnetFeaturePipeline(self.feat_info)
            self.asr.set_input_pipeline(feat_pipeline)
            self.asr.init_decoding()
            data = wav.data()[0]
            # print(data.data[0:2048])
            # print(data.dim)
            # print(data[0:2048])
            # print(data[100:120])
            last_chunk = False
            # part = 1
            prev_num_frames_decoded = 0
            for i in range(0, len(data), self.chunk_size):
                if i + self.chunk_size >= len(data):
                    last_chunk = True
                feat_pipeline.accept_waveform(wav.samp_freq, data[i : i + self.chunk_size])
                # print(data[i : i + self.chunk_size])
                if last_chunk:
                    feat_pipeline.input_finished()
                self.asr.advance_decoding()
                num_frames_decoded = self.asr.decoder.num_frames_decoded()
                if not last_chunk:
                    if num_frames_decoded > prev_num_frames_decoded:
                        prev_num_frames_decoded = num_frames_decoded
                        out = self.asr.get_partial_output()
                        # print(key + "-part%d" % part, out["text"])
                        print(out["text"])
                        # part += 1
            self.asr.finalize_decoding()
            out = self.asr.get_output()
            # print(key + "-final", out["text"])
            print(out["text"])

    def infer2(self, data_chunk):
        data = SubVector(data_chunk)
        # print(len(data.data))
        # print(data.data)
        # print(wav)
        feat_pipeline = OnlineNnetFeaturePipeline(self.feat_info)
        self.asr.set_input_pipeline(feat_pipeline)
        self.asr.init_decoding()
        # data = wav.data()[0]
        # print(data.dim)
        # print(type(data))
        last_chunk = False
        # part = 1
        prev_num_frames_decoded = 0
        for i in range(0, len(data), self.chunk_size):
            if i + self.chunk_size >= len(data):
                last_chunk = True
            feat_pipeline.accept_waveform(16000.0, data[i : i + self.chunk_size])
            if last_chunk:
                feat_pipeline.input_finished()
            self.asr.advance_decoding()
            num_frames_decoded = self.asr.decoder.num_frames_decoded()
            if not last_chunk:
                if num_frames_decoded > prev_num_frames_decoded:
                    prev_num_frames_decoded = num_frames_decoded
                    out = self.asr.get_partial_output()
                    # print(key + "-part%d" % part, out["text"])
                    print(out["text"])
                    # part += 1
        self.asr.finalize_decoding()
        out = self.asr.get_output()
        # print(key + "-final", out["text"])
        print(out["text"])
        return out['text']

    def infer_chunk(self, data_chunk, feat_pipeline, last_chunk):
        data_chunk = SubVector(data_chunk)
        print(data_chunk)
        if data_chunk.is_zero():
            print("data is zero")
        # print(data_chunk.data)
        # print(type(data_chunk))
        # print(data_chunk.dim)
        feat_pipeline.accept_waveform(16000.0, data_chunk)
        if last_chunk:
            feat_pipeline.input_finished()
        self.asr.advance_decoding()
        if not last_chunk:
            out = self.asr.get_partial_output()
            return out["text"]
        return ""


if __name__ == "__main__":
    kaldi = KaldiInfer()
    kaldi.infer_in()
    # print(output['alignment'])

# Decode (whole utterance)
# for key, wav in SequentialWaveReader("scp:wav.scp"):
#     feat_pipeline = OnlineNnetFeaturePipeline(feat_info)
#     asr.set_input_pipeline(feat_pipeline)
#     feat_pipeline.accept_waveform(wav.samp_freq, wav.data()[0])
#     feat_pipeline.input_finished()
#     out = asr.decode()
#     print(key, out["text"], flush=True)

# Decode (chunked + partial output)
