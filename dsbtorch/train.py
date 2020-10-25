# Training function for DeepSponsorBlock
# --------------------------------------

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
	best_dev_acc = 0

	optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_acc = train_for_epoch(model, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
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
		# How do we generate minibatches?
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

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
    dev_acc = compute_accuracy(predict(dev_data))  # TODO: Implement these!
    print("- dev accuracy: {:.2f}".format(dev_acc * 100.0))
    return dev_acc