
from model import *
from dataloader import *
from torch.utils.data import DataLoader
from sklearn import preprocessing
import torch.optim as optim
import time

def train(model, train_dataloader, criterion, optimizer, device, le):
      # Train one epoch
      model.train()
      train_loss = 0
      train_correct = 0
      for batch_idx, (inputs, targets) in enumerate(train_dataloader):
          #inputs, targets = inputs.to(device), targets.to(device)
          targets = torch.tensor(le.transform(targets))
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()

          train_loss += loss.item()
          train_correct += (outputs.argmax(1) == targets).sum().item()

      train_loss /= len(train_dataloader.dataset)
      train_acc = train_correct / len(train_dataloader.dataset)
      return train_loss, train_acc

def evaluate(model, test_dataloader, criterion, device, le):
      model.eval()
      test_loss = 0
      test_correct = 0
      for batch_idx, (inputs, targets) in enumerate(test_dataloader):
          #inputs, targets = inputs.to(device), targets.to(device)
          targets = torch.tensor(le.transform(targets))
          outputs = model(inputs)
          loss = criterion(outputs, targets)

          test_loss += loss.item()
          test_correct += (outputs.argmax(1) == targets).sum().item()

      test_loss /= len(test_dataloader.dataset)
      test_acc = test_correct / len(test_dataloader.dataset)
      return test_loss, test_acc

def main():
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model', type=str, default='resnet', help='Model architecture (resnet, regnet, efficientnet, vit)')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone architecture (resnet50, regnet, efficientnet, vit)')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights for backbone')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save trained model')

    args = parser.parse_args()

    #TODO
    #Add hflip, rotation, color transforms,...
    transforms = Compose([Resize((224, 224)),
                      ToTensor()])
    datasets = FurnitureDataset(args.data_dir, transforms).get_datasets()

    train_dataloader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=True)

    #TODO
    #Modify dataloader and include self.classes as attribute
    #target should be only numerical
    labels = []
    for label in os.listdir(data_dir):
        if not label.startswith('.'):
        labels.append(label)
    print(labels)

    le = preprocessing.LabelEncoder()
    le.fit(labels)

    # Define hyperparameters
    #TODO
    #move hyperparams to .yml file
    batch_size = args.batch_size
    learning_rate = arg.lr
    num_epochs = args.num_epochs

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FurnitureModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        best_acc = 0.0
        # Train one epoch
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device, le)
        #Test
        test_loss, test_acc = evaluate(model, train_dataloader, criterion, device, le)

        # Save best model weights
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_weights.pt')

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"val_loss={test_loss:.4f} "
                f"val_acc={test_acc:.4f} "
                f"time={time.time()-start_time:.2f}s")


