# HDTransformer

Transformer Implementation for Early Hotspot Detection in power lines using BOTDR frequency data.

## Description

This is the implementation of the model for detecting hot spots in power-lines using data from a BOTDR.



### Abstract

  The early detection of power line hot spots is crucial to prevent catastrophic failures and blackouts. In this paper, we propose a novel approach to classify power line hot spots using BOTDR frequency measurements from optical fibers parallel to the power lines. We use transformers, a state-of-the-art deep learning architecture, to automatically learn features from the frequency data and perform the classification task.
Our approach to solving this problem starts with combining the raw frequency data with positional embeddings to provide the lacking sequence information to the transformer.
In addition, we also introduce a class token to the input sequence that captures information from the input sequence. 
We then use a transformer-based model to learn the spectral features and by using an MLP at the exit end of the transformer to classify the power line hot spots. 
To evaluate the performance of our approach, we conduct experiments on both an experimental dataset that simulates the change of temperature on the power lines and a real-world dataset of frequency data collected from optical fibers parallel to the power lines. 

  Moreover, we perform a sensitivity analysis to investigate the impact of various factors, such as the size of the transformer model, the number of spectral features, and the amount of training data, on the classification performance. Our analysis reveals that the performance of our approach is robust to these factors, and the optimal configuration depends on the specific characteristics of the dataset.
Overall, our approach provides a promising solution for the early detection of power line hot spots using frequency data of parallel optical fibers. 
Using transformers enables us to automatically learn the relevant features from the data and achieve high classification accuracy.
Our proposed models manage a 99% accuracy on experimental data and 91% on real-world data.
