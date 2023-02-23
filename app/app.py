from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
#import torch.nn as nn
#import torchvision.models as models
from

class BackboneModel(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        self.name = name
        self.pretrained = pretrained

        if name.startswith('resnet'):
            self.model = models.resnet50(pretrained=pretrained)
            self.num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()  # remove the last classification layer
        else:
            raise ValueError(f"Unsupported backbone model: {name}")

    def forward(self, x):
        return self.model(x)

class FurnitureModel(nn.Module):
    def __init__(self, backbone_name = 'resnet'):
        super().__init__()

        # Load the backbone model
        self.backbone = BackboneModel(backbone_name)
        # Add a new fully connected layer with 3 output neurons
        self.fc = nn.Linear(self.backbone.num_features, 3)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


app = Flask(__name__)

# Load the trained model
model_path = "best_weight.pth"
model = FurnitureModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define image transformations same as training
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']
    image = Image.open(image_file)

    # Transform the image
    image_tensor = image_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)

    #TODO
    #Get folder names and sort to make it sys invariant
    labels = ['Bed', 'Chair', 'Sofa']
    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        class_name = labels[prediction]

    # Return the prediction
    return jsonify({'class': class_name})

if __name__ == '__main__':
    app.run(debug=True)