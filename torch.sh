sudo docker run \
--gpus all \
--rm -it \
--name=mitsuo \
-p 8888:8887 \
-v $(pwd):/workdir \
-w /workdir \
torch \

bash
