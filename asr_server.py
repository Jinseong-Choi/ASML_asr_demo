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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class ASR(asr_pb2_grpc.ASRServiceServicer):
    def __init__(self):
        # Initialize and load the model and pipeline here, so it's done only once
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32

        model_path = "/home/ubuntu/Workspace/g_dino_serve/models"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        processor = AutoProcessor.from_pretrained(model_path)
        self.pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, torch_dtype=self.torch_dtype)

    def transcribe_speech(self, filepath, language):
        output = self.pipe(
            filepath,
            max_new_tokens=256,
            generate_kwargs={
                "task": "transcribe",
                "language": language,
            },  # update with the language you've fine-tuned on
            chunk_length_s=30,
            batch_size=1,
        )
        return output["text"]

    def Transcribe(self, request, context):
        # request.audio_data = audio
        # request.sampling_rate = sr
        # request.language = language
        transcription = self.transcribe_speech(request.filepath, request.language)
        reply = asr_pb2.ASRReply()
        reply.transcription = transcription
        return reply


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
