import matplotlib.pyplot as plt


def training_epochs(model, num_epochs, loss_function, optimizer, train_dataloader):
    losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            print(f"{i} \ {len(train_dataloader)}", end="\r")
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #labels = torch.nn.functional.one_hot(labels)
            #print(inputs.shape, labels)

            # opt.zero_grad()
            # loss.backward()
            # opt.step()
            # print(inputs)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            #print(outputs.shape, labels.shape)
            loss = loss_function(outputs, labels)
            # print(outputs, labels, loss)
            # print("\n\n")
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print_cycle_length = 100
            # if i % print_cycle_length == print_cycle_length-1:
            #     print(
            #         f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1)}')
            #     running_loss = 0.0
        average_loss = running_loss/len(train_dataloader)
        print(average_loss)
        losses.append(average_loss)
    plt.plot(losses)
    plt.savefig("./plot_losses")

    print('Finished Training')

    return model
