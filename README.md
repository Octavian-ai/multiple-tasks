
# Learning multiple tasks with gradient descent

This is the companion code for [this medium article](https://medium.com/p/23447735519b/edit) and
can be run on [FloydHub](https://www.floydhub.com/davidmack/projects/mnist-forget).

This code sets up a convolutional network that is sequentially trained on a series of different tasks. By varying which tasks are trained against and how long is spent on each task I've been able to investigate which factors improve the network's test accuracy on the different tasks.

## Preliminary results

This is a small first step into investigating multi-task learning. I've copied the abstract from [the accompanying article](https://medium.com/p/23447735519b/edit), it has a lot more charts and detail explaining these observations:

> Learning multiple tasks is an important requirement for artificial general intelligence. I apply a basic convolutional network to two different tasks. The network is shown to exhibit successful multi-task learning. The frequency of task switchover is shown to be a crucial factor in the rate of network learning. Stochastic Gradient Descent is shown to be better at multi-task learning than Adam. Narrower networks are more forgetful.

## Experimental setup

I’m using a very simple network for these experiments:
- Input: 3 channel image, 28x28, training batches of 100
- Two hidden layers: 1 convolutional layer, 1 fully connected layer
- ReLU activation, max pooling, batch normalisation and dropout are used
- Output: Softmax classifier over 10 classes
- Stochastic Gradient Descent is used to train the network with a learning rate of 0.1


## Running the code

You can run this code locally (`pipenv install` will set up dependencies, then `./train.py`) or on FloydHub (`./floyd-run.sh`).

Here's an example of running the code:
`
python task.py \
		--log-dir /output \
		--data-dir /data-mnist/MNIST_data --data-dog-dir /data-dog \
		--gpu \
		--task-time 2000 \
		--max-steps 30000 \
		--input-style combined \
	 	--tasks dog
`

- **task-time** How many training steps to perform on each task
- **max-steps** Total number of training steps to perform
- **input-style** There are a couple of architecture options for how the training data is injested
- **tasks** A list of tasks to run. You can choose between:
  - **dog** Recognise dog breeds
  - **id** Standard MNIST
  - **ref** MNIST, reflected on one axis
  - **inv** MNIST, with pixels inverted (E.g. black becomes white)
  - **noise** Random noise as the x input, MNIST labels as the y target
  - **perm** MNIST with the rows permutated. The permutation is stable across training samples/cycles
  - **zero** Zeros as x input, MNIST labels as the y target
  - **one** Ones as the x input, MNIST labels as the y target




