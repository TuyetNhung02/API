import os
import logging
import grpc

import chatbot_service_pb2
import chatbot_service_pb2_grpc
from pipeline import ChatBot
from concurrent import futures

chatbot_interface = os.environ.get('CHATBOT_SERVER_INTERFACE', '0.0.0.0')
chatbot_port = int(os.environ.get('CHATBOT_SERVER_PORT', 5002))

class ChatServiceServicer(chatbot_service_pb2_grpc.ChatServiceServicer):
    
    def __init__(self, data_path):
        self.generator = ChatBot(data_path)
        

        
    def AnswerGenerate(self, request, context):
        result = self.generator.GenerateAnswer(request.message)
        return chatbot_service_pb2.AnswerResponse(message=result)
    
def serve():
    logging.info("Server starting ...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=32))
    chatbot_service_pb2_grpc.add_ChatServiceServicer_to_server(
        ChatServiceServicer(data_path=data_path),
        server
    )
    server.add_insecure_port('{}:{}'.format(chatbot_interface, chatbot_port))
    server.start()
    logging.info(f"Started server on {chatbot_interface}:{chatbot_port}")
    server.wait_for_termination()
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    serve()
    