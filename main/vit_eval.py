import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import ViT, AudioDataset

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

PATH = "./vit-ckpts/vit-model-1689780787-29.pt"

checkpoint = torch.load(PATH, map_location="cpu")
model = ViT(n_classes=checkpoint['num_of_classes'])
model = nn.DataParallel(model)
model.load_state_dict(checkpoint['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

epoch = checkpoint['epoch']
loss = checkpoint['loss']

root = "../data/spectograms-1/test"
train_dataset = AudioDataset(root, classes=checkpoint["classes"], transform=transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)
model.cuda()
print("num of classes", checkpoint["num_of_classes"])
print("classes:", checkpoint["classes"])

y_true = []
y_pred = []

counter = 0

for inputs, emitters, labels in tqdm.tqdm(train_loader):
    # if counter == 3:
    #     break
    inputs = inputs.to(device)
    emitters = emitters.to(device)
    labels = labels.to(device)
    outputs = model(inputs, emitters)
    _, preds = torch.max(outputs, 1)
    y_true += list(labels)
    y_pred += list(preds)
    counter += 1


y_true = [y.to("cpu") for y in y_true]
y_pred = [y.to("cpu") for y in y_pred]

c = 0
print("y_pred:")
print(y_pred)
print("y_true:")
print(y_true)

for i in range(len(y_true)):
    if y_pred[i] == y_true[i]:
        c += 1

print(c/len(y_true))


cm = confusion_matrix(y_true, y_pred)
# cm = cm / cm.sum(axis=1)[:, None]
print(cm.shape)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=checkpoint["classes"])
disp.plot()
plt.show()
