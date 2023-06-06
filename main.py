import torch
import numpy as np

from DataEmbedding import DataEmbedding
from Encoder import Encoder
from EncoderLayer import EncoderLayer
from MultiHeadAttention import MultiHeadAttention
from PositionalEmbedding import PositionalEmbedding
from TokenEmbedding import TokenEmbedding
from Transformer import Transformer


# accepts data in the shape of [x, 96, 256] (x is the batch)
# will return an array with length x
# values are between 0 and 1
#   label 0 is normal
#   label 1 is abnormal
# the closer to 0, or 1 the more sure the model is 
def make_prediction(data, model):
    data = torch.Tensor(data).float().cuda()
    _, classes = model(data)
    classes = classes.detach().cpu().numpy().squeeze(1).squeeze(1)
    return classes


# example class
def main():
    # path to the model
    path = '/content/drive/MyDrive/torch_load_model.pth'

    model = torch.load(path)
    model.eval()
    
    test_data = np.random.rand(500,96,256)
    print(test_data.shape)


    predictions = make_prediction(test_data, model)
    print(predictions)
    print("predictions complete")

if __name__ == "__main__":
    main()

