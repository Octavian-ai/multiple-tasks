#!/bin/bash

floyd run --tensorboard --gpu --env tensorflow-1.3:py2 \
	--data redeipirati/datasets/mnist/1:/data-mnist \
	--data davidmack/datasets/dogbreeds/1:/data-dog \
	"python task.py \
		--log-dir /output \
		--data-dir /data-mnist/MNIST_data --data-dog-dir /data-dog \
		--gpu \
		--task-time 2000 \
		--max-steps 30000 \
		--input-style combined \
	 	--tasks id dog"
