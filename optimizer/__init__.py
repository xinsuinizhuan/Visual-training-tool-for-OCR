import torch.optim as optim


def build_optimizer(model, lr=0.0001):
    return optim.Adam(model.parameters(), lr, betas=(0.5, 0.999), weight_decay=0.001)


def build_scheduler(optimizer, step_size=60, gamma=0.5):
    scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    return scheduler