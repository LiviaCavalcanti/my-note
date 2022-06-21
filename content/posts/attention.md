---
title: "Attention"
date: 2022-03-24T21:42:56-03:00
draft: true
---

BERT is a model that became a reference in the NPL field. It uses the well-knew attention technique to perform different language tasks.

## Previous Models

Before BERT, the standard language models were unidirectional. That is, the token can only attend to the previous tokens, what “could be very harmful when applying finetune-based approaches to token-level tasks”.

These unidirectional models are called sequence to sequence (seq2seq). They consist of stacked RNN layers and work well with Long Short-term Memory (LSTM) and Gated Recurrent Unit (GRU) components. 

Due to the bottleneck problem, seq2seq models work well only for small sequences, with less than 20 tokens.
- Bottleneck Problem

    The computation is done on a _reference window_, that is, it is necessary to look at the symbols that occured in past to capture the sentence context. In theory, the reference window of any length is possible. The current RNNs implementations, though, tend to forget information that are “many” timesteps far behind.
    
    In a seq2seq model, there is a weight to control the retention of information at every step. But, as the model process futher untis of data, the amount of information stored from earlier steps vanishes, similarly to the gradient vanishing problem.


There are two approaches that share the same objective function during pre-training. They use unidirectional language models to learn general language representation:
- feature-based approach (ELMo) uses task-specific architectures that include the pre-trained representations as additional features.
- fine-tuning approach (OpenAI GPT)  introduces minimal task-specific parameters and is trained on the downstream tasks by simply fine-tuning all pretrained parameters


## Attention 
The main process unit for BERT is called attention:
> **Attention is a content-based retrieval mechanism** that produces/absorbs data based on how similar it is to every position in its memory. **It works by comparing all the content** represented in the form of word embeddings, for instance, and each word is re-represented **using a weighted combination of its neighborhood** and thus summarizing the content.

Traditional neural networks already have an implicit attention mechanism. Networks trained to object recognition tasks, for instance, tend to give more attention to the most relevent parts of an image.

There are many ways of doing attention. The most popular one was proposed by Bahdanau et al. where attention is a set of trainable weights that can be tuned using the standard backpropagation algorithm.

In theory, attention is defined as the weighted average of values. The weighting is a learned function which means that it depends on the data.

An network that uses attention can be made by an encoder and a decoder part. The encoder gets the input and transforms it in an encoded representation. The encoder can be a [bidirectional RNN](https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn).

The decoder task is to trasform the encoded requesentation into a target output that can be used to accomplish some task. 

In terms of mathematical representation, considering $s$ a hidden state, $t$ the state index and $c$ a context vector, the attention representation of the decoder part is 

$s_t=f(s_{t-1}, y_{t-1},c_t)$

where:

- $y$ is output a target sequence of length $m$

- $c_t$ is a context vector of the sum of hidden states of the input sequence $x$ until the state $s$,

Since $c_t$ is data-dependent, it needs a notion of memory. Also, it is trainable and given by the function:

$c_t=\sum_{i=1}^{n} \alpha_{t,i}h_i$

Note that, $c_t$ is calculated for every $t$ position of the output sequence $y$ and its calculation depends on all $n$ positions of the input $x$

### Self-attention

The self-attention mechanism add context information from other data points[tokens] in the same sentence. It is analogous to the retrieval operation in a database or in a dictionary. 

The input data point is represented as a query. The self-attention maps a query to a set of key-value pairs. To retrieve the value that suits the best to the query, self-attention compute similarity between the query and all the keys. A common method used in transformer is dot product.

The final result is given by a weighted-sum between the just-calculated attention scores and the tokens values. 

The key tokens do not have to contemplate the full sentence. But, in most implementations, they do.

Since the resulting value is calculated using all the tokens, attention ignores the distance between the tokens has within a text.

![model](/home/livia/Imagens/attention.png)

## Transformer model

Transformer is multi-layer encoder-decoder architecture. Both enconder and decoder are composed of identical sub-units called layers that might be stacked according to the desired depth. The number of layers of encoder-decoder is denoted by $N$.

The first step to use the Transformer model is to prepare the input. Every input sentence has to be split into sets of tokens where the arrangement of the elements does not matter. 

Then, the tokens are coded into a representation that is able to be used by the model. Transformer uses word embeddings representation that is a distributed low-dimensional space of continuous-valued vectors. So, an original sentence is represented as a vector of embeddings of its tokens, for which the mathematical operations are appliable and carry the original setence meaning with it, even in its results.

### Layers

The Transformer layers are made by two main elements Multi-head attention and Feed-Foward

**Attention Layer**

The main transformer's processing unit is the the self-attention layer. This layer's output is a contextualized vector and can be used as input of another self-attention layer.

It is based on linear transformations, since every position of this layer's input is vectors projected into in the key [K], value [V] and query [Q] spaces, following the idea exposed earlier. Then, it is calculated what is called by the authors **Scaled Dot-Product Attention**. 

In the Transformer self-attention layer, though, the calculations are more than the attention itself. Going through more details, for every query, the following steps are performed::

1. Dot-product: calculates the dot product of the query with all the keys.
2. Scaling: divides the result of the previous step by a scalar factor - $\sqrt{d_k}$, where $d_k$ is the dimension of the keys space. It aims to normalize dot-products by not letting them blowing up
3. Softmax: obtains weights from the scaled vectors. The scaling previous step helps to push softmax to small gradients regions. Also, dropout is applied.
4. Second dot-product: the weights from softmax are mixed with the value vector.

The output vector, for each query, is a weighted sum of the attention weights and the sentence vectors. And the self-attention weights must be learned from the word embeddings of the input data.


**Multi-head Attention**

To allow the model to find different relationships in the input information, in the Transformer, the keys, values and queries are linearly projected into different representation subspaces with dimensions $d_k$, $d_v$ and $d_q$, respectively. Each projection, called *head*, is learned and happens $h$ times for every layer. 

The attention is performed in parallel for all heads in a same layer, their final results are concatenated and, once again, linear projected. 

The dimension of the heads is reduced to small dimensions so the FLOPS are kept the same to the attention in a full dimensionality.

**Feed-forward**

It is made of two linear transformations applied to every position with ReLu activation in between. Each layer has its own parameters for the linear transformation.


Obs.: Every Self-attention outputs and FF layers outputs are followed by a residual layer and a Normalization layer. 
Residual connections carry position information to every layer, helping in the distribution of the inter-model maps of distribution and accuracy. While the Normalization helps the model convergence.

![Considering a sentence, it is possible to represent it by a vector of embeddings of its tokens. This vector can be multiplied by a matrix of weights. Then, their results can suffer a dot product operation and generate scores that are normalized by a softmax layer, that lays the values in [0,1] maximizing the non-zero values. These values will be used to re-weight the sentence embeddings. All the re-weighted vectors will be summed up, resulting in a weighted sum. The output will be a vector built from the embeddings and trained weights.](BERT%207ac7c/Untitled.png)


### Model structure
Wrapping up the encoder and the decoder architecture based on what we saw of the layers:

**Encoder**

It's formed by N layers of

- Multi-head attention layer
- Normalization
- Feed-Forward
- Normalization

In this order.

The input of the encoder, the text itself, is transformed into embeddings and added a positional encoding, to differentiate the same words in the same sentence.

**Decoder**

The decoder layers has one more attention sub-layer than the encoder. It's formed by N layers of

- Masked Multi-head attention 
- Normalization
- Multi-Head Attention
- Normalization
- Feed-forward layer
- Normalization

The input of the decoder is the output text generated so far in the task and the encoder output. As the encoder input, the output text also represented as embeddings. It is offset by one position and masked, so the decoder can't look ahead to the unknown text. 

The Multi-Head Attention combines the last encoder output with the contextualized output text from the Masked Multi-head attention sublayer


The decoder output is linear transformed and goes through a softmax layer to calculate the output probabilities.

## Transformer advantages


- Smaller path for long-range dependencies
- not have to deal with gradient vanish/explosion
- smaller training steps
- enable parallel computation
- it models language hierarchy that is important for language models
- it is not restricted to memory because does not depend on a finite representation in-memory


