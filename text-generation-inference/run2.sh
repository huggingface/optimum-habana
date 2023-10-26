model=bigscience/bloom-560m
volume=/software/data/pytorch/bloom/bloom-weights/ # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi_lt --model-id $model --revision ac2ae5fab2ce3f9f40dc79b5ca9f637430d24971 --sharded true --num-shard 2  2>&1 | tee log2.txt
