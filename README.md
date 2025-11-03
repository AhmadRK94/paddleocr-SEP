### For running the cpu version
- run `docker build -t fastapi-paddle-cpu -f Dockerfile-cpu .`
- then run `docker run --rm -p 8000:8000 fastapi-paddle-cpu`
- then open `127.0.0.1:8000/docs` and upload your image.


### For running the gpu version
- run `docker build -t fastapi-paddle-gpu -f Dockerfile-gpu .`
- then run `docker run --rm --gpus all -p 8000:8000 fastapi-paddle-gpu`
- then open `127.0.0.1:8000/docs` and upload your image.