# makemore
An autoregressive character-level language model for making more things

## Introduction

## Learnings

- bigram: look at the given character, predict the next one (always only look at two characters)
	- predict by counting the occurances of two characters after each other and choose the next one based on multinomial probabilities
	- achieved nll of roughly 2.46
- making it a neural net:
	the outputs of the neural net for input 0 (=> '.')
	we transfored the '.' to an integer (0)
	one-hot encoded it as a 27d vector
	fed to neural net by multiplying with random weights W
	exponentiated to get outputs between 0 and 1
	normalized over rows (in total outputs for each input have probability of 1) to get probabilities
	neural net now predicts probability for each of the 27 next possible characters after '.'
	then we can calculate the loss (neg log likelihood) on predicting the next real character
	```
	probs[0]
	tensor([0.0069, 0.0547, 0.0351, 0.0310, 0.0081, 0.0422, 0.0129, 0.0688, 0.0444,
	0.0223, 0.0168, 0.0233, 0.0776, 0.0435, 0.0523, 0.1066, 0.0867, 0.0229,
	0.0771, 0.0102, 0.0176, 0.0472, 0.0159, 0.0392, 0.0113, 0.0115, 0.0139])
	```
		--------
	bigram example 1: .e (indexes 0,5)
	input to the neural net: 0
	output probabilities from the neural net: tensor([0.0607, 0.0100, 0.0123, 0.0042, 0.0168, 0.0123, 0.0027, 0.0232, 0.0137,
			0.0313, 0.0079, 0.0278, 0.0091, 0.0082, 0.0500, 0.2378, 0.0603, 0.0025,
			0.0249, 0.0055, 0.0339, 0.0109, 0.0029, 0.0198, 0.0118, 0.1537, 0.1459])
	label (actual next character): 5
	probability assigned by the net to the the correct character: 0.012286253273487091
	log likelihood: -4.3992743492126465
	negative log likelihood: 4.3992743492126465
	--------
	```
	# gradient descent
for k in range(100):

	#forward pass
	xenc = F.one_hot(xs, num_classes=27).float()	#0, 5, 13, 13, 1
	logits = (xenc @ W)	# predict log-counts shape[5, 27]: 27 predictions for each of the 5 inputs
	counts = logits.exp()
	probs = counts / counts.sum(1, keepdims=True)	#probabilities for next character
	loss = -probs[torch.arange(num), ys].log().mean()
	print(loss.item())

	#backward pass
	W.grad = None	#more efficient than setting to 0
	loss.backward()

	#update
	W.data += -100 * W.grad
	```
	- achieved nll of roughly 2.46, which is equal to with counting (because bigrams are so simple)
	- but now we can actually extend to do the forward pass on multiple characters


## Definitions to remember
- Keeping Dimensions: Ensures that the resulting tensor from the sum operation has the same shape as the original tensor, making element-wise operations straightforward.
- Broadcasting: A mechanism that allows element-wise operations on tensors of different shapes by automatically expanding the smaller tensor to match the shape of the larger tensor.
- one-hot encoding: take a vector with the length of the amount of integers you want to represent, and put a 0 at all numbers that's not the integer, and a 1 where it is; example: [0, 0, 1, 0, 0] = 2; needed to feed ints into neural nets (you can't just plug in integers)
- log likelihood:
	likelihood of a model = the product of all probabilities of all bigrams
	will be a very tiny number, so you use log-likelihood (will be between -inf and 0)
	log(a * b * c ..) = log(a) + log(b) + log(c) + ..



