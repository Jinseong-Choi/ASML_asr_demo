import asyncio
import logging

import grpc
from infer_pb2 import ImgReply, ImgRequest
from infer_pb2_grpc import InferenceStub
import mmcv

fn = '/home/ubuntu/src/mmdetection/demo/dogs.webp'
data = mmcv.imread(fn)
dim_data = data.shape
color_map = 'bgr'
async def run() -> None:
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = InferenceStub(channel)
        request = ImgRequest()
        request.data = data.tobytes()
        request.height  = dim_data[0]
        request.width = dim_data[1]
        request.text = '사람. 개.'
        request.color_map = color_map
        request.threshold = 0.3
        response = await stub.infer(request)
    print('response', response)

if __name__=="__main__":
    logging.basicConfig()
    asyncio.run(run())

