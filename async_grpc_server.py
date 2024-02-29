import asyncio
import logging

import grpc
from infer_pb2 import ImgReply
from infer_pb2 import ImgRequest
from infer_pb2_grpc import add_InferenceServicer_to_server
from infer_pb2_grpc import InferenceServicer

from mmdet.apis import DetInferencer
import numpy as np

_cleanup_coroutines = []

class Inferencer(InferenceServicer):
    def __init__(self) -> None:
        super().__init__()
        model_name = '/home/ubuntu/src/mmdetection/configs/grounding_dino/grounding_dino_r50_scratch_8xb2_1x_coco_kor.py'
        checkpoint = '/home/ubuntu/src/mmdetection/checkpoints/grounding_dino_r50_scratch_8xb2_1x_coco_kor/epoch_12.pth'
        device = 'cpu'
        self.inferencer = DetInferencer(model_name, checkpoint, device)

    async def infer(self, request: ImgRequest, context: grpc.aio.ServicerContext) -> ImgReply:
        data_len = len(request.data)
        assert data_len == (request.width * request.height * 3)

        img_data = np.frombuffer(request.data, dtype=np.uint8).reshape((request.height, request.width, 3))
        if request.color_map == 'rgb':
            img_data = img_data[:,:,::-1]

        result = self.inferencer(img_data, texts=request.text, pred_score_thr=request.threshold, return_vis=False, show=False,  no_save_vis=True, draw_pred=False, custom_entities=True)
        scores = np.array(result['predictions'][0]['scores'])
        labels = np.array(result['predictions'][0]['labels'], dtype=np.int32)
        bboxes = np.array(result['predictions'][0]['bboxes'])
        detected_idx = np.where(scores > request.threshold)
        response = ImgReply()
        response.num_object = len(detected_idx[0])
        response.scores.extend(scores[detected_idx])
        response.labels.extend(labels[detected_idx])
        response.bboxes.extend(bboxes[detected_idx].reshape((-1,)))
        #print(scores[detected_idx].shape, labels[detected_idx].shape, bboxes[detected_idx].reshape((-1,)).shape)
        #print(response)

        return response 


async def serve() -> None:
    server = grpc.aio.server( options = [
        ('grpc.max_send_message_length', 12_000_000), 
        ('grpc.max_receive_message_length', 12_000_000) ]
        )
    add_InferenceServicer_to_server(Inferencer(), server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)

    logging.info("Starting server on %s", listen_addr)
    await server.start()
    async def server_graceful_shutdown():
        logging.info("Starting graceful shutdown...")
        await server.stop(5)

    _cleanup_coroutines.append(server_graceful_shutdown())

    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve())
    finally:
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
