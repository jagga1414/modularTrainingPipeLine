import torch


def train_epoch(model, train_dataloader, optimizer, config):
    i = 0
    total = len(train_dataloader)
    for bd in train_dataloader:
        output = model(input_ids=bd.input_ids.to(config.device), 
                      attention_mask=bd.attention_mask.to(config.device), 
                      labels=bd.labels.to(config.device))
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("loss", loss, f"{i}/{total}")
        i += 1


def train(model, train_dataloader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08)
    for epoch in range(config.max_epoch):
        train_epoch(model, train_dataloader, optimizer, config)
