import torch
import numpy as np

from DataEmbedding import DataEmbedding
from Encoder import Encoder
from EncoderLayer import EncoderLayer
from MultiHeadAttention import MultiHeadAttention
from PositionalEmbedding import PositionalEmbedding
from TokenEmbedding import TokenEmbedding
from Transformer import Transformer



def make_prediction(data, model):
    """
    accepts data in the shape of [x, 96, 256] (x is the batch)
        will return an array with length x,
        values are between 0 and 1,
        label 0 is normal,
        label 1 is abnormal,
        the closer to 0, or 1 the more sure the model is \n

    returns: array with length X, where X is analogous to 'x' (number of samples) 
    """

    data = torch.Tensor(data).float().cuda()
    _, classes = model(data)
    classes = classes.detach().cpu().numpy().squeeze(1).squeeze(1)
    return classes

def prepare_data_format(path):
    """
    reads a file where each row is a single point in the fiber.
    each row has 96x256 columns (96 temporal measurements, 256 frequencies per temporal measurement).\
    \n
    returns: numpy array in the shape of [x, 96, 256] where "x" is the total number of rows in the original file.

    """
    with open(path) as file:
        Lines = file.readlines()
        points = []

        for line in Lines:
            arr = line.split(';')
            points.append(arr)
        
    d = np.array(points, dtype=float)
    data = d.reshape((-1, 96, 256))

    return data

# example usage
def main():
    path = 'torch_load_model.pth'
    model = torch.load(path)
    model.eval()
    
    processed_data = prepare_data_format(path='testing_data.txt')
    
    predictions = make_prediction(processed_data, model)
    print(predictions)

    print("predictions complete")

if __name__ == "__main__":
    main()

