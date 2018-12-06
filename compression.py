#Ivan Dimitrov
#Compression Techniques 
import torch
import numpy as np

def examine_model(model):
	num_child = 0 

	for child in model.children():
		print ("Child: ", num_child) 
		print (child)	
		num_child +=1


def examine_parameter_structure(model):
	num_child = 0 

	for child in model.children():
		print ("Child: ", num_child) 
		print (child)	
		num_child +=1

def conv_weights_structure(model):
	num_child = 0 

	for child in model.children():
		if num_child == 4:
			print (child)
			for param in child.parameters(): 
				print (param)
				print ("Size of Param", param.size())
		num_child +=1


def prune_model(model):
	num_child = 0 

	for child in model.children():
		if num_child == 4:
			for param in child.parameters(): 
				temp_size = list(param.size())
				# print (temp_size)
				if temp_size == [1024]:
					#print (param)
					simple_prune(param, 1024)
					#print (param)
		num_child +=1	

def simple_prune(param, n):
	print ("in simple prune")
	#param.data[param.data < 0] = 0
	acc = 0
	for i in range(n):
		if param.data[i] < .01 and param.data[i] > .01:
			param.data[i] = 0
			acc+=1
	print ("Acc is: ", acc)


def collect_stats(model, collect_parama):
	num_child = 0 
	t = 0 
	for child in model.children():
		if num_child == 4:
			for param in child.parameters(): 
				print (t, flush=True)
				t+=1
				temp_size = list(param.size())
				if (len(temp_size) == 1):
					for i in range(temp_size[0]): 
						collect_parama.append(param.data[i])

				else:
					for i in range(temp_size[0]):
						for j in range(temp_size[1]):
							for k in range(temp_size[2]):
								collect_parama.append(param.data[i][j][k])
				if t == 20:
					return collect_parama
		num_child +=1	
	return collect_parama


def full_prune(model, threshold):
	num_child = 0 
	num_of_parameters = 0
	changed_params = 0
	for child in model.children():
		if num_child == 4:

			num_of_parameters += sum(p.numel() for p in child.parameters() if p.requires_grad)
			for param in child.parameters(): 

				param.data[np.absolute(param.data) < threshold] = 0
				changed_params += (param.numel() - param.nonzero().size(0))
		num_child +=1	
	print ("Total Parameters: ", num_of_parameters)
	print ("Changed Parameters: ", changed_params)


def layer_prune(model, threshold, layers):
	""" Each conv block as weights + bias, that why we pretend there are 30 layers when really there are 15 in this model
	"""
	num_child = 0 
	num_of_parameters = 0
	changed_params = 0
	for child in model.children():
		if num_child == 4:
			curr_place = 0
			num_of_parameters += sum(p.numel() for p in child.parameters() if p.requires_grad)
			for param in child.parameters(): 
				if curr_place in layers:
					param.data[np.absolute(param.data) < threshold] = 0
					changed_params += (param.numel() - param.nonzero().size(0))
				curr_place+=1
		num_child +=1	
	print ("Total Parameters: ", num_of_parameters)
	print ("Changed Parameters: ", changed_params)



def mask_generation(model, threshold):
	num_child = 0 

	for child in model.children():
		if num_child == 4:
			curr_place = 0
			for param in child.parameters(): 
				param.data[np.absolute(param.data) < threshold] = 0
				temp_ten = param.clone()
				temp_ten.detach()
				temp_ten[temp_ten != 0] = 1
				print (curr_place)
				print (param)
				print (param.size())
				if curr_place%2 == 0:
					model.convolutions[curr_place//2].bias_mask = temp_ten
				else:
					model.convolutions[curr_place//2].weight_mask = temp_ten
				curr_place +=1

		num_child +=1	







def binarize_weights(model):
	"""doesn't actually binarize -> but a side effect since almost all weights [-1, 1]
	"""
	num_child = 0
	encoder = None
	ecoder = None 
	for child in model.children():
		if num_child == 0:
			encoder = child
		elif num_child == 1:
			decoder = child
		num_child+=1 

	for child in encoder.children():
		for param in child.parameters():
			param.data = torch.tensor(torch.tensor(param.data, dtype=torch.short), dtype=torch.float).cuda()
	for child in decoder.children():
		for param in child.parameters():
			param.data = torch.tensor(torch.tensor(param.data, dtype=torch.short), dtype=torch.float).cuda()



def half_point_floating_point(model):
	num_child = 0
	encoder = None
	decoder = None 
	for child in model.children():
		if num_child == 0:
			encoder = child
		elif num_child == 1:
			decoder = child
		num_child+=1 

	num_child = 0
	for child in encoder.children():
		num_child += 1
		for param in child.parameters():
			param.data = torch.tensor(param.data, dtype=torch.half, requires_grad=True).cuda()

	num_child = 0
	for child in decoder.children():
		for param in child.parameters():
			param.data = torch.tensor(param.data, dtype=torch.half, requires_grad=True).cuda()
