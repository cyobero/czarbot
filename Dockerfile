FROM nvidia/cuda:10.2-base
COPY . ./czarbot

RUN apt-get update && apt-get install \
    python3 
    
