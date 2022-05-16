import torch


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sample in test_loader:
            unicode_features = sample['matrix'].to(device)
            euclidean_coordinates = sample['coordinates'].to(device)
            # geo_country_code = sample['geo_country_code'].to(device)

            # Task 1 - country code prediction
            # test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

            # Task 2 = MvMF loss
            target = torch.zeros(len(euclidean_coordinates)).to(device)
            output = model(unicode_features, euclidean_coordinates)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
