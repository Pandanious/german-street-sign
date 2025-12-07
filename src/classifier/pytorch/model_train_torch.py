import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
from pre_process_images_torch import pre_process
from models_torch import s_custom_model,GTSRBModel,LTSModel

epoch = 30

# class -> model as argument, values at model selection at __init__
def train(num_epochs = epoch, lr = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds,val_ds, _ = pre_process(batchsize = 64)
    num_classes = train_ds.dataset.num_classes
    #model = GTSRBModel(num_classes).to(device)
    #model = s_custom_model(num_classes,60,60).to(device)
    model = LTSModel(num_classes,60,60).to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs, eta_min=1e-5)
    #optimizer = torch.optim.Adam(model.parameters(), lr = lr,)
    #checkpoint_path = "/home/panda/projects/german-street-sign/models/pytorch/gtsrb_model_without_softmax.pt"
    #checkpoint_path = "/home/panda/projects/german-street-sign/models/pytorch/custom_model_scheduler_weight_decay_without_softmax.pt"
    checkpoint_path = "/home/panda/projects/german-street-sign/models/pytorch/LTSM_model.pt"
    output_metric = "/home/panda/projects/german-street-sign/models/pytorch/LTSM_model.txt"
    #output_metric = "/home/panda/projects/german-street-sign/models/pytorch/gtsrb_model_without_softmax.txt"
    #output_metric = "/home/panda/projects/german-street-sign/models/pytorch/custom_model_scheduler_weight_decay_without_softmax.pt.txt"
    metric_log = []

    for epoch in range(num_epochs):
        print("Starting_Training")
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for inputs, labels in train_ds:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criteria(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            scheduler.step()

        train_loss = running_loss / total
        train_acc = running_correct / total

        print("starting_validation")
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_ds:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criteria(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        e_summary = (f"Epoch {epoch+1}/{num_epochs} "
                      f"| train_loss {train_loss:.4f} acc {train_acc:.4f} "
                      f"| val_loss {val_loss:.4f} acc {val_acc:.4f}")
        print(e_summary)
        metric_log.append(e_summary)


    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_metric).parent.mkdir(parents=True, exist_ok=True)
    print("Saving_checkpoint")
    torch.save(model.state_dict(), checkpoint_path)
    print("Model should be saved under", checkpoint_path)
    print("Writing metrics to txt file under",output_metric)
    Path(output_metric).write_text("\n".join(metric_log)+"\n")
    return model


if __name__ == "__main__":
    train()
