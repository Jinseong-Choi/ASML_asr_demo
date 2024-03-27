# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC asr.Transcribe server."""

from concurrent import futures
import logging

import grpc
import asr_pb2
import asr_pb2_grpc

import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.generation.configuration_utils import GenerationConfig

import cProfile
import pstats

import os
import math
import re
import time
import editdistance

from faster_whisper import WhisperModel

chunk_size = 3000

def remove_non_korean(sentence):
    cleaned_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]', '', sentence)
    return cleaned_sentence

def get_audio_duration_torchaudio(filepath):
    audio_tensor, sample_rate = torchaudio.load(filepath)
    duration = audio_tensor.shape[1] / sample_rate
    return duration

def calculate_cer(ref, hyp):
    hyp = remove_non_korean(hyp)
    ref = remove_non_korean(ref)
    cer = editdistance.eval(ref.replace(" ", ""), hyp.replace(" ", "")) / len(ref)
    return cer

def chunkfy(audio, stride=2800):
    assert stride <= chunk_size
    B, C, T = audio.size()

    # audio: torch.tensor of shape (1, T, C)
    num_chunks_process = 1 + (T - chunk_size) // stride
    len_padded = num_chunks_process * stride + chunk_size
    padding_needed = len_padded - audio.size(-1)
    audio = F.pad(audio, (0, padding_needed), "constant", 0)
    num_chunks_process += 1
    
    # batchfy input
    B, C, T = audio.size()
    if stride == chunk_size:
        audio = audio.transpose(0, -1).view(num_chunks_process, -1, C, 1).squeeze(-1)  # B, T, C
    else:
        audio = audio.squeeze(0).transpose(0,1) # T, C
        segments = []
        for idx in range(num_chunks_process):
            start = idx * stride
            end = start + chunk_size
            segment = audio[start:end, :]
            segments.append(segment)
        audio = torch.stack(segments, dim=0)
    audio = audio.transpose(1, 2) # B, C, T

    return audio

class fast_pipeline():
    def __init__(self, model):
        self.model = model
        self.epd_margin = 0.1

    def __call__(self, filepath, generate_configs):
        if "condition_on_prev_tokens" in generate_configs:
            generate_configs["condition_on_previous_text"] = generate_configs.pop("condition_on_prev_tokens")
        if "logprob_threshold" in generate_configs:
            generate_configs["log_prob_threshold"] = generate_configs.pop("logprob_threshold")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Start timing for generating
        start_event.record()
        segments, _ = self.model.transcribe(filepath, beam_size=1, without_timestamps=False, **generate_configs)
        texts = []
        for segment in segments:
            if float(segment.start) >= float(audio_duration) - self.epd_margin:
                break
            texts.append(segment.text)
            print(segment)
        exit()
        transcription = "".join(texts)
        end_event.record()
        torch.cuda.synchronize()
        total_processing_time = start_event.elapsed_time(end_event) / 1000

        RTF = total_processing_time / audio_duration

        print(f"Total processing time: {total_processing_time} seconds")
        print(f"Real Time Factor (RTF): {RTF}")

        return transcription

class gen_pipeline():
    def __init__(self, processor, model, audio_pipe, device, chunkfy=False):
        self.processor = processor
        self.model = model.to(device)
        self.audio_pipe = audio_pipe
        self.device = device
        self.chunkfy = chunkfy

        self.generation_configs = GenerationConfig.from_pretrained("models")

    def __call__(self, filepath, generate_configs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Start timing for preprocessing
        start_event.record()
        audio = self.audio_pipe.preprocess(filepath, chunk_length_s=0)
        for aud in audio:
            inputs = aud["input_features"]
        end_event.record()
        torch.cuda.synchronize() # Wait for the events to be recorded!
        preprocessing_time = start_event.elapsed_time(end_event)

        # chunkfy data and prepare inputs
        if inputs.shape[-1] > chunk_size and self.chunkfy:
            inputs = chunkfy(inputs)
        inputs = {"input_features": inputs.to(self.device, torch.float32), "generation_config": self.generation_configs, **generate_configs}

        # Start timing for model forwarding
        start_event.record()
        generated_ids = self.model.generate(**inputs)
        end_event.record()
        torch.cuda.synchronize()
        model_forward_time = start_event.elapsed_time(end_event)

        # Start timing for decoding
        start_event.record()
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        transcription = transcription[0] if len(transcription) == 1 else "".join(transcription)
        end_event.record()
        torch.cuda.synchronize()
        decoding_time = start_event.elapsed_time(end_event)

        # Calculate the total processing time and RTF as before
        total_processing_time = (preprocessing_time + model_forward_time + decoding_time) / 1000.0 # Convert milliseconds to seconds
        RTF = total_processing_time / audio_duration

        print(f"Preprocessing time: {preprocessing_time / 1000.0} seconds")
        print(f"Model forwarding time: {model_forward_time / 1000.0} seconds")
        print(f"Decoding time: {decoding_time / 1000.0} seconds")
        print(f"Total processing time: {total_processing_time} seconds")
        print(f"Real Time Factor (RTF): {RTF}")

        return transcription

class ASR():
    def __init__(self, pipe_type='fast-pipe'):
        # Initialize and load the model and pipeline here, so it's done only once
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32
        self.pipe_type = pipe_type

        if self.pipe_type == 'fast-pipe':
            # model_path = "/home/ubuntu/Workspace/gradio_asr/faster-whisper-Klec"
            # self.model = WhisperModel(model_path, device="cuda", compute_type="float16")
            self.model = WhisperModel('large-v3', device="cuda", compute_type="float16")
        else:
            model_path = "/home/ubuntu/Workspace/gradio_asr/models"
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            self.model.to(self.device)
            processor = AutoProcessor.from_pretrained(model_path)        

        if self.pipe_type == 'hf-pipe':
            # transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline object
            self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, torch_dtype=self.torch_dtype)
        elif self.pipe_type == 'gen-pipe':
            pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, torch_dtype=self.torch_dtype)
            self.pipe = gen_pipeline(processor=processor, model=self.model, audio_pipe=pipe, device=self.device)
        elif self.pipe_type == 'fast-pipe':
            self.pipe = fast_pipeline(model=self.model)

    def transcribe_speech(self, filepath, language):
        if self.pipe_type == 'hf-pipe':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            output = self.pipe(
                filepath,
                max_new_tokens=256,
                generate_kwargs={
                    "task": "transcribe",
                    "language": language,
                },  # update with the language you've fine-tuned on
                chunk_length_s=0,
                batch_size=1,
            )

            end_event.record()
            torch.cuda.synchronize() # Wait for the events to be recorded!
            # Calculate the total processing time and RTF as before
            total_processing_time = start_event.elapsed_time(end_event) / 1000.0
            RTF = total_processing_time / audio_duration

            print(f"Total processing time: {total_processing_time} seconds")
            print(f"Real Time Factor (RTF): {RTF}")

            return output["text"]
        else:
            temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            if language == "ko":
                no_speech_threshold = 0.55
                logprob_threshold = -0.3
            else:
                no_speech_threshold = 0.6
                logprob_threshold = -1.0

            generate_config = {"task": "transcribe", "language": language}
            if audio_duration > 30:
                generate_config["max_new_tokens"] = 256
                generate_config["condition_on_prev_tokens"] = False
                generate_config["no_speech_threshold"] = no_speech_threshold
                generate_config["temperature"] = temperature
                generate_config["logprob_threshold"] = logprob_threshold

            output = self.pipe(
                filepath,
                generate_config
            )
            return output

if __name__ == "__main__":
    asr = ASR(pipe_type='fast-pipe')
    
    # fname = "/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Validation/D01/E01/S000028/000011.wav"
    # fname = "/home/ubuntu/Workspace/gradio_asr/examples/songs/byg.mp3"
    fname = "/home/ubuntu/Workspace/gradio_asr/examples/animals/SLAAO21000001.wav"
    # fname = "/home/ubuntu/Workspace/DB/korean_db/korean_asr_db/KlecSpeech/Training/D12/G02/S001024/000143.wav"
    audio_duration = get_audio_duration_torchaudio(fname)

    tname = fname.replace(".mp3", ".txt") if "mp3" in fname else fname.replace(".wav", ".txt")
    with open(tname, 'r') as f:
        ref = f.read()

    hyp = asr.transcribe_speech(fname, "ko")
    
    with open(tname.replace(".txt", "_hyp.txt"), "w") as f:
        f.write(hyp)

    print(calculate_cer(ref, hyp))