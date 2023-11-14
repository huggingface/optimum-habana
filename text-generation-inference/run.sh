model=bigscience/bloom-560m
volume=/software/data/pytorch/bloom/bloom-weights/ # share a volume with the Docker container to avoid downloading weights every run

docker run -p 8082:80 -td --name sarkar_tgi -v $volume:/data -v /home/sasarkar/:/sarkar --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host tgi_gaudi_ss 2>&1 | tee log.txt
