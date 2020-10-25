# Training for DeepSponsorBlock
# --------------------------------------
import numpy as np
import torch.nn as nn
import skimage
import sklearn.metrics as metrics

def train(model, train_data, dev_data, output_path, minibatch_size=32, n_epochs=10, lr=0.001):
	"""
	Trains a model for multiple epochs

	@param model (nn.Module): A model from models.py
	@param train_data
	@param dev_data
	@param output_path (str): A path to save the params of the best-performing model
	@param minibatch_size (int): The number of video clips in a minibatch
	@param n_epochs (int): The number of epochs to train for
	@param lr (float): Learning rate for Adam

	@return
	"""
	best_dev_f_score = 0

	optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_f_score = train_for_epoch(model, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_f_score > best_dev_f_score:
            best_dev_f_score = dev_f_score
            print("New best dev accuracy! Saving model.")
            torch.save(model.state_dict(), output_path)
        print("")

def train_for_epoch(model, train_data, dev_data, optimizer, loss_fn, minibatch_size):
	"""
	Trains a model for a single epoch (i.e. once over the entire training data)

	@return dev_acc (float): The model's accuracy on the dev set after this epoch
	"""
	model.train()
	n_minibatches = len(train_data) / minibatch_size  # depends on which DataSet we use
	loss_meter = AverageMeter()

	with tqdm(total=(n_minibatches)) as prog:
		# TODO: How do we iterate over minibatches?
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here

            output = model(train_x)
            loss = loss_fn(output, train_y)
            parser.model.zero_grad()
            loss.backward()
            optimizer.step()

            prog.update(1)
            loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    model.eval()
    preds = model(dev_data)
    dev_labels = None  # TODO: How to get labels?
    dev_f_score = compute_weighted_f1(preds, dev_labels)
    print("- dev weighted F1: {:.2f}".format(dev_f_score * 100.0))
    return dev_f_score

def compute_f_score(preds, labels, beta2=0.25):
	"""
	Computes the F-beta score for the given set of predictions
	"""
	preds = preds > 0.5
	precision = metrics.precision_score(labels, preds)
	recall = metrics.recall_score(labels, preds)
	f_score = (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall)
	return f_score