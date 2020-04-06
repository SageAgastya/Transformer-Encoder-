# Transformer-Encoder-
This is a pytorch implementation of Transformer encoder which can be used for classification purpose. 
Here, I have assumed a particular type of input format as mentioned in the code.
The padding is done beautifully taking care of max_words, max_sentences and dim_size.
A pad object does three tasks:
      > Calls embedder's callable object to get elmo representation.
      > Callable object Positional Encoder is created and added to the input representation.
      > Does padding.
Creating an Encoder will do all the subsequent tasks.
An Encoder object consists of multiple transformer encoder layers.
Categorical Cross-Entropy Loss is minimized to learn the classifier.
