from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import FurnitureModel

app = Flask(__name__)

# Load the trained model
model_path = "model.pth"
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