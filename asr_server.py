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
import protos.asr_pb2 as asr_pb2
import protos.asr_pb2_grpc as asr_pb2_grpc

import torch
from faster_whisper import WhisperModel

class fast_pipeline():
    def __init__(self, model):
        self.model = model
        self.epd_margin = 0.1

    def __call__(self, filepath, generation_config, duration):
        if "condition_on_prev_tokens" in generation_config:
            generation_config["condition_on_previous_text"] = generation_config.pop("condition_on_prev_tokens")
        if "logprob_threshold" in generation_config:
            generation_config["log_prob_threshold"] = generation_config.pop("logprob_threshold")

        segments, _ = self.model.transcribe(filepath, beam_size=1, without_timestamps=False, **generation_config)

        notimestamped_text = ""
        timestamped_text = ""
        for segment in segments:
            if float(segment.start) >= float(duration) - self.epd_margin:
                break
            notimestamped_text += f"{segment.text}\n"
            timestamped_text += f"[{segment.start:.2f}:{segment.end:.2f}] {segment.text}\n"

        return notimestamped_text, timestamped_text

class ASR(asr_pb2_grpc.ASRServiceServicer):
    def __init__(self):
        # Initialize and load the model and pipeline here, so it's done only once
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32
        self.use_model("fast_v3")

        # Decoding configurations
        temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        no_speech_threshold = 0.6
        logprob_threshold = -1.0

        self.generate_config = {"task": "transcribe"}
        self.generate_config["max_new_tokens"] = 256
        self.generate_config["condition_on_prev_tokens"] = False
        self.generate_config["no_speech_threshold"] = no_speech_threshold
        self.generate_config["temperature"] = temperature
        self.generate_config["logprob_threshold"] = logprob_threshold

    def use_model(self, model_choice):
        model_path = f"/home/ubuntu/Workspace/gradio_asr/models/{model_choice}"
        self.model = WhisperModel(model_path, device="cuda", compute_type="float16")
        self.pipe = fast_pipeline(model=self.model)

    def transcribe_speech(self, filepath, language, with_timestamps, duration):
        if language == "ko":
            self.generate_config["no_speech_threshold"] = 0.55
            self.generate_config["logprob_threshold"] = -0.3
        else:
            self.generate_config["no_speech_threshold"] = 0.6
            self.generate_config["logprob_threshold"] = -1.0
        self.generate_config["language"] = language

        notimestamped_text, timestamped_text = self.pipe(
            filepath,
            self.generate_config,
            duration
        )
        
        return timestamped_text if with_timestamps else notimestamped_text

    def Transcribe(self, request, context):
        transcription = self.transcribe_speech(request.filepath, request.language, request.with_timestamps, request.duration)
        reply = asr_pb2.ASRReply()
        reply.transcription = transcription
        return reply
    
    def ReloadModel(self, request, context):
        try:
            self.use_model(request.model_name)
            return asr_pb2.ReloadModelReply(success=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Failed to reload model: {e}')
            return asr_pb2.ReloadModelReply(success=False)

def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    asr_service = ASR()  # Create an instance of the ASR service
    asr_pb2_grpc.add_ASRServiceServicer_to_server(asr_service, server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
