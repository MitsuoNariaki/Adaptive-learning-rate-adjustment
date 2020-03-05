sudo docker run \
--gpus all \
--rm -it \
--name=username \
-p 8888:8888 \
-v $(pwd):/workdir \
-w /workdir \
torch \
"$@"
