import torch


def train(
    model,
    device,
    train_loader,
    criterion,
    optimizer,
    epoch,
    train_loss,
    train_acc,
    mse,
):
    model.train()
    curr_loss = 0
    t_pred = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        output, recons = model(images)
        loss_m = criterion(output, targets)
        vae = mse(recons, images)
        loss = loss_m + vae  # +model.encoder.kl

        loss.backward()
        optimizer.step()

        curr_loss += loss.sum().item()
        _, preds = torch.max(output, 1)
        t_pred += torch.sum(preds == targets.data).item()

        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} vae_Loss {:.6f}".format(
                    epoch,
                    batch_idx * len(images),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss_m.item(),
                    vae.item(),
                )
            )

            train_loss.append(loss.sum().item() / len(images))
            train_acc.append(preds.sum().item() / len(images))
    epoch_loss = curr_loss / len(train_loader.dataset)
    epoch_acc = t_pred / len(train_loader.dataset)

    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)

    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch_loss,
            t_pred,
            len(train_loader.dataset),
            100.0 * t_pred / len(train_loader.dataset),
        )
    )

    return train_loss, train_acc, epoch_loss


def valid(model, device, test_loader, criterion, epoch, valid_loss, valid_acc, mse):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            output, recons = model(images)
            loss_m = criterion(output, targets)
            vae = mse(recons, images)
            loss = loss_m + vae  # +model.encoder.kl

            test_loss += loss.sum().item()  # sum up batch loss

            _, preds = torch.max(output, 1)
            correct += torch.sum(preds == targets.data)

            if batch_idx % 10 == 0:
                print(
                    "Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} vae_Loss {:.6f}".format(
                        epoch,
                        batch_idx * len(images),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        loss_m.item(),
                        vae.item(),
                    )
                )

                valid_loss.append(loss.sum().item() / len(images))
                valid_acc.append(preds.sum().item() / len(images))

    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_acc = correct / len(test_loader.dataset)

    valid_loss.append(epoch_loss)
    valid_acc.append(epoch_acc.item())

    print(
        "\nValid Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return valid_loss, valid_acc
