import torch
import random
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt # for making figures


def main():
	random.seed(42)

	# hyperparameters
	block_size = 8 # context length: how many characters do we take to predict the next one?
	n_embd = 24 # dimensions of character embeddings
	n_hidden = 68 # number of hidden neurons
	steps = 30000 # number of training steps
	batch_size = 32 # batch size

	# load dataset
	words = load_words("./names.txt")
	stoi, itos, vocab_size = build_vocab(words)

	n1 = int(0.8*len(words))
	n2 = int(0.9*len(words))
	train  = build_dataset(words[:n1], stoi, block_size=8)     # 80%
	dev = build_dataset(words[n1:n2], stoi, block_size=8)   # 10%
	test  = build_dataset(words[n2:], stoi, block_size=8)     # 10%

	# create model
	model = create_model(n_embd, n_hidden, vocab_size)
	# train model
	optimized = optimize(model, train, steps, batch_size)

	# evaluate model
	evaluate(optimized, train, dev, test)
	# sample
	sample(optimized, itos, block_size)

def load_words(file):
	words = open(file, 'r').read().splitlines()

	return words

def build_vocab(words):
	# build the vocabulary of characters and mappings to/from integers
	chars = sorted(list(set(''.join(words))))
	stoi = {s:i+1 for i,s in enumerate(chars)}
	stoi['.'] = 0
	itos = {i:s for s,i in stoi.items()}
	vocab_size = len(itos)

	return stoi, itos, vocab_size

def build_dataset(words, stoi, block_size=5):
	X, Y = [], []

	for w in words:
		context = [0] * block_size
		for ch in w + '.':
			ix = stoi[ch]
			X.append(context)
			Y.append(ix)
			context = context[1:] + [ix] # crop and append

	X = torch.tensor(X)
	Y = torch.tensor(Y)

	return X, Y

class Linear:

	def __init__(self, fan_in, fan_out, bias=False):
		self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5	#draw weights randomly from gaussian and normalize with sqrt(inputs)
		self.bias = torch.zeros(fan_out) if bias else None

	def __call__(self, x):
		self.out = x @ self.weight
		if self.bias is not None:
			self.out += self.bias
		return self.out

	def parameters(self):
		return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1D:

	def __init__(self, dim, eps=1e-5, momentum=0.1):
		self.eps = eps
		self.momentum = momentum
		self.training = True	#to use bnmean during training and bnrunning for
		# parameters (trained with backprop)
		self.gamma = torch.ones(dim)
		self.beta = torch.zeros(dim)
		# buffers (trained with running momentum update)
		self.running_mean = torch.zeros(dim)
		self.running_var = torch.ones(dim)

	def __call__(self, x):
		# forward pass
		if self.training:
			if x.ndim == 2:
				dim = 0
			elif x.ndim == 3:
				dim = (0, 1)
			xmean = x.mean(dim, keepdim=True) # batch mean
			xvar = x.var(dim, keepdim=True) # batch variance
		else:
			xmean = self.running_mean
			xvar = self.running_var
		xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
		self.out = self.gamma * xhat + self.beta
		# update buffers
		if self.training:
			with torch.no_grad():
				self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
				self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
		return self.out

	def parameters(self):
		return [self.gamma, self.beta]

class Tanh:

	def __call__(self, x):
		self.out = torch.tanh(x)
		return self.out

	def parameters(self):
		return []

class Embedding:

	def __init__(self, num_embeddings, embedding_dim):
		self.weight = torch.randn((num_embeddings, embedding_dim))	#number of chars to be embedded, number of dims used

	def __call__(self, IX):
		self.out = self.weight[IX]
		return self.out

	def parameters(self):
		return [self.weight]

class FlattenConsecutive:

	def __init__(self, n):
		self.n = n

	def __call__(self, x):
		B, T, C = x.shape
		x = x.view(B, T//self.n, C*self.n)
		if x.shape[1] == 1:
			x = x.squeeze(1)
		self.out = x
		return self.out

	def parameters(self):
		return []

class Sequential:

	def __init__(self, layers):
		self.layers = layers

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)
		self.out = x
		return self.out

	def parameters(self):
		#get parameters of all layers and stretch them into one list
		return [p for layer in self.layers for p in layer.parameters()]

def create_model(n_embd, n_hidden, vocab_size):

	model = Sequential([
		Embedding(vocab_size, n_embd),
		FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
		FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
		FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
		Linear(n_hidden, vocab_size),
	])

	with torch.no_grad():
		#last layer: make less confident, to stabilize loss
		model.layers[-1].weight *= 0.1

	parameters = model.parameters()
	print("Model built.\nNumber of params:", sum(p.nelement() for p in parameters))	#number of parameters in total
	for p in parameters:
		p.requires_grad = True

	return model

def optimize(model, data, max_steps, batch_size):
	lossi = []
	Xtr, Ytr = data

	# Initialize variables for the progress bar
	bar_length = 40
	start_time = time.time()

	print("\nOptimization started:")
	for i in range(max_steps):

		# Minibatch construct
		ix = torch.randint(0, Xtr.shape[0], (batch_size,))
		Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

		# Forward pass
		logits = model(Xb)
		loss = F.cross_entropy(logits, Yb) # get loss

		# Backward pass
		for p in model.parameters():
			p.grad = None
		loss.backward()

		# Update
		lr = 0.1 if i < (0.75*max_steps) else 0.01 # step learning rate decay
		for p in model.parameters():
			p.data += -lr * p.grad

		# Track stats
		lossi.append(loss.log10().item())

		# Update progress bar
		if i % 100 == 0 or i == max_steps - 1:
			progress = i / max_steps
			block = int(round(bar_length * progress))
			elapsed_time = time.time() - start_time
			text = f"\r[{i}/{max_steps}] [{'#' * block + '-' * (bar_length - block)}] " \
				   f"Loss: {loss.item():.4f} | Time: {elapsed_time:.2f}s"
			print(text, end='')

		# Print detailed stats occasionally
		if i % 10000 == 0 or i == max_steps - 1:
			print(f'\nStep {i}/{max_steps}: Loss = {loss.item():.4f}')

	print("\nOptimization finished.\n")
	return model


def evaluate(model, train, val, test):
	#put layers into eval mode
	for layer in model.layers:
		layer.training = False
	# evaluate the loss
	@torch.no_grad() # this decorator disables gradient tracking: tells pytorch to not track gradients, making it more efficient
	def split_loss(split):
		x,y = {
			'train': train,
			'val': val,
			'test': test,
		}[split]
		logits = model(x)
		loss = F.cross_entropy(logits, y) # get loss
		print(split, "split:" , f"{loss.item():.3f}")

	print("Evaluation:")
	split_loss('train')
	split_loss('val')
	split_loss('test')


def sample(model, itos, block_size=5):
	print("\n\nSampling:")
	# sample from the model
	for _ in range(20):

		out = []
		context = [0] * block_size # initialize with all ...
		while True:
			# forward pass
			logits = model(torch.tensor([context])) # (1,block_size,d)
			probs = F.softmax(logits, dim=1)
			ix = torch.multinomial(probs, num_samples=1).item()
			context = context[1:] + [ix]
			out.append(ix)
			if ix == 0:
				break

		print(''.join(itos[i] for i in out))

if __name__ == "__main__":
	main()
