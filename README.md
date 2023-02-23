# furniture-classification-api
Simple API with Flask to serve a classification model from Pytorch
This is a project to classify furniture images into three categories: bed, sofa, and chair. The project uses a pretrained ResNet model as the backbone and fine-tunes it on the dataset.


## Requirements
- Python 3.x
- PyTorch 1.x
- torchvision
- numpy
- scikit-learn

* You can install the required packages using the following command: *
```
pip install -r requirements.txt
```
## Training
* To train the model, run the following command: *
```
python models/train.py --data_dir /data --backbone resnet --batch_size 16 \
--num_epochs 10 --out_dir ../app
```

## Running prediction api
* One can execute the flask prediction api
```
python python app/app.py
```
* And test it with 

```
python app/test/py
```

## Additionaly you can download the weights:
Copy an paste the following [best_weights](https://drive.google.com/file/d/1uMHvBxqMN7pxaX3FlYwuiSt4cyAVM64J/view?usp=share_link) weights into the </> app/folder </>