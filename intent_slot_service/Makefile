GRPC_SOURCES = chatbot_service_pb2.py chatbot_service_pb2_grpc.py

all: $(GRPC_SOURCES)

$(GRPC_SOURCES): chatbot_service.proto
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. chatbot_service.proto

clean:
	rm $(GRPC_SOURCES)