import os
import sys
import json

import grpc
import chatbot_service_pb2
import chatbot_service_pb2_grpc

from underthesea import word_tokenize

MONITOR_SERVER_INTERFACE = os.environ.get('HOST', 'localhost')
MONITOR_SERVER_PORT = int(os.environ.get('PORT', 5002))

CHANNEL_IP = f"{MONITOR_SERVER_INTERFACE}:{MONITOR_SERVER_PORT}"

def main():
    channel = grpc.insecure_channel(CHANNEL_IP)
    stub = chatbot_service_pb2_grpc.ChatServiceStub(channel)
    message = 'Cho em hỏi học phí ngành công nghệ phần mềm CMU'
    message = word_tokenize(message, format="text")
    result = stub.AnswerGenerate(chatbot_service_pb2.QuestionRequest(message=message)) #goisever
    print(json.loads(result.message))
    
if __name__ == "__main__":
    main()