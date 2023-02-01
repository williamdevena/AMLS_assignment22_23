import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def training_epochs(model, device, num_epochs, loss_function,
                    optimizer, train_dataloader, test_dataloader,
                    train_val_dataloader, val_dataloader):
    losses = []
    test_accuracies = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print(f"Epoch: {epoch}")
        for i, data in enumerate(tqdm(train_dataloader), 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = torch.squeeze(outputs).to(device)
            loss = loss_function(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        test_acc = testing(model=model, device=device,
                           test_dataloader=test_dataloader)
        train_acc = testing(model=model, device=device,
                            test_dataloader=train_val_dataloader)
        val_acc = testing(model=model, device=device,
                          test_dataloader=val_dataloader)

        average_loss = running_loss/len(train_dataloader)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        val_accuracies.append(val_acc)
        losses.append(average_loss)
        print("--------")
        print(f"Loss: {average_loss}")
        print(f"Train Accuracy: {train_acc}")
        print(f"Val Accuracy: {val_acc}")
        print(f"Test Accuracy: {test_acc}")
        print("--------")

    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss function")
    plt.savefig("./plot_loss")
    plt.close()

    plt.plot(train_accuracies, label="Train")
    plt.plot(val_accuracies, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("./plot_validation")
    plt.close()

    print('Finished Training')

    return model


def testing(model, device, test_dataloader, binary):
    correct = 0
    for i, data in enumerate(tqdm(test_dataloader), 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs).to(device)
        output = output.float()
        label = labels.float()

        if binary:
            output = 1 if output >= 0.5 else 0
        else:
            output = torch.unsqueeze(torch.argmax(output), dim=0)

        if output == label:
            correct += 1

    accuracy = correct/len(test_dataloader)

    return accuracy


def pred(model, device, test_dataloader):
    """


    Args:
        model (_type_): _description_
        test_dataloader (_type_): _description_
    """

    pred = []
    for i, data in enumerate(tqdm(test_dataloader), 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs).to(device)
        output = output.float()
        label = labels.float()

        output = 1 if output >= 0.5 else 0
        pred.append(output)

    return torch.tensor(pred)
