# Notes for Week1 Sequencial model
**author:** Lu KONG  
**course site:** [here](https://www.coursera.org/learn/nlp-sequence-models/home/welcome)  
**Professor:** Andrew Ng

**Outline:**
- [Notes for Week1 Sequencial model](#notes-for-week1-sequencial-model)
- [Recurrent Neural Networks](#recurrent-neural-networks)
  - [1. Introduction](#1-introduction)
    - [1.1 Examples of sequence data:](#11-examples-of-sequence-data)
    - [1.2 Notation](#12-notation)
    - [1.3 How to represent the words in the sentense](#13-how-to-represent-the-words-in-the-sentense)
  - [2. Standard network:](#2-standard-network)
  - [3. Recurrent Neural Network](#3-recurrent-neural-network)
    - [3.1 Composition](#31-composition)
    - [3.2 Problem](#32-problem)
    - [3.3 Forword Propagation](#33-forword-propagation)
      - [Compress parameters](#compress-parameters)
    - [3.4 Backpropagation through time](#34-backpropagation-through-time)
    - [3.5 Different types of RNNs](#35-different-types-of-rnns)
  - [4. Language model and sequence generation](#4-language-model-and-sequence-generation)
    - [4.1 Language modelling with an RNN](#41-language-modelling-with-an-rnn)
    - [4.2 Sampling novel sequences](#42-sampling-novel-sequences)
  - [5. Problems with RNN](#5-problems-with-rnn)
    - [5.1 Exploding gradients with RNN](#51-exploding-gradients-with-rnn)
    - [5.2 Vaninshing gradients with RNN](#52-vaninshing-gradients-with-rnn)
  - [6. Long Short Term Memory (LSTM)](#6-long-short-term-memory-lstm)
  - [7. Introduction to some other mentioned RNNs](#7-introduction-to-some-other-mentioned-rnns)
    - [7.1 Bidirectional RNN](#71-bidirectional-rnn)
    - [7.2 Deep RNNs](#72-deep-rnns)

# Recurrent Neural Networks

## 1. Introduction

### 1.1 Examples of sequence data:
input or(and) output in form of sequence

- speech recognition
- Music generation
- Sentiment classification
- DNA sequence analysis
- Machine translation
- Video activity recognition
- Name entity recognition

![](w1_000.png)
  
### 1.2 Notation
- to possition `i` of the sequence `x`, we denote:  
  $$x^{<i>}$$
- the same for the output `y`:
  $$y^{<i>}$$
- for $i^{\text{th}}$ training example, we denote the $t^{\text{th}}$ position of the input and output as:
  $$x^{(i)<t>}\\
  y^{(i)<t>}$$ 

- We denote the length of the sequence as a stopping time:
  $$T_x^{(i)}\\  T_y^{(i)}$$
  , which denote the stopping time of the $i^{\text{th}}$ input `x` and output `y`

### 1.3 How to represent the words in the sentense
1. Firstly we need our `vocabulary` or `dictionary`
2. Use a `one-hot` presentation to represent a `word` in a vecter:  
   `0` for all except `1` for the sequencial number of the `word`
3. `token` for unknown word


## 2. Standard network:  
Network has $T_x^{(i)}$ inputs and $T_y^{(i)} = T_x^{(i)}$ outputs `correspond` with each other.

Problems:  
- Inputs, outputs can be different lengths in different examples.
- Doesn't share feathers learned across differents of texts. (`too independent`)
    
  We hope that the model can share its experience: When first learnt a word as a `nom`, it will have some experience when it see it in other position that this word could be a `nom`. This is also for reduce the number of parameters

## 3. Recurrent Neural Network

### 3.1 Composition
`Instead of` only using just the input of this position, the RNN will take the `activations` of `previous position` as inputs together to predict the output of the current position.

Add vector of zeros as $a^{<0>}$ so that the first position works as laters.

We can denote parameters of this network in three types:  

- $\omega_{ax}$ : for parameters work with input $x^{<i>}$
- $\omega_{aa}$ : for parameters work with input $a^{<i-1>}$
- $\omega_{ya}$ : for parameters works with activation $a^{<i>}$ to get $\hat{y}^{<i>}$ as output.

### 3.2 Problem 
- Problem:  It looks only these parameters of previous positions `but not later positions`, which is also very important `linguisticly` in natural language.

- Solution: Bidirectional RNN (`BRNN`)

### 3.3 Forword Propagation
![](w1_001.png)

#### Compress parameters
Originally the fomules wrote as follow :
$$
\begin{aligned}
    a^{<t>} &= g(W_{aa}a^{<t-1>} + W_{ax}x^{<t>} + b_a)\\
    \hat{y}^{<t>} &= g(W_ya^{<t>} + b_y)
\end{aligned}
$$

By stacking the inputs vectors $a^{<t-1>}$ and $x^{<t>}$ we get the input  
$$\begin{bmatrix}
    a^{<t-1>}\\
    x^{<t>}
\end{bmatrix}$$
note as ${\mathop{}^A_X}^{<t>}$.

And the parameters $W_{aa}$ and $W_{ax}$ are transfered as:
$$
W_a = 
\begin{bmatrix}
    W_{aa} &W_{ax}
\end{bmatrix}$$

So that,

$$
\begin{aligned}
    W_a {\mathop{}^A_X}^{<t>} &= 
\begin{bmatrix}
    W_{aa} &W_{ax}
\end{bmatrix}
\begin{bmatrix}
    a^{<t-1>}\\
    x^{<t>}
\end{bmatrix} \\
&= W_{aa}a^{<t-1>} + W_{ax}x^{<t>}
\end{aligned}\\
$$

and we rewrite the fomulations as :  
$$
\begin{aligned}
    a^{<t>} &= g( W_a {\mathop{}^A_X}^{<t>} + b_a)\\

    \hat{y}^{<t>} &= g(W_ya^{<t>} + b_y)
\end{aligned}
$$ 

### 3.4 Backpropagation through time

**important :** former parameters infect all later activations by recurrenting

**Loss function：**

$$
\begin{aligned}
    \mathcal{L}^{<t>}(\hat{y}^{<t>}, {y}^{<t>}) &= - y^{<t>} \log{\hat{y}^{<t>}} - (1-y^{<t>}) \log(1-\hat{y}^{<t>})\\
    \mathcal{L}(\hat{y}^{<t>}, {y}^{<t>}) &=\sum_{t=1}^{T_y}\mathcal{L}^{<t>}(\hat{y}^{<t>}, {y}^{<t>})
\end{aligned}
$$
*Backpropogation through time*

### 3.5 Different types of RNNs

There existes cases where $T_x \not ={} T_y$.

- `Many` to `Many`
- `Many` to `One`  
  ex : Sentiment classification
- `One` to `Many`  
  ex: Music generation
- `One` to `One`  
  Just the normal problem without sequence.

The most often used structure is of cause : `Many` to `Many`

For example, when we do `machine translation`, sentences in different languages can have different length.  

## 4. Language model and sequence generation
How to build a language model

Speech recognition: give every sentence a probability

Output only sentence that is likely right.

### 4.1 Language modelling with an RNN

- Traning set : Large corpus of English text.
  Tokenize
  - `<EOS>` token : End of Sentense 
  - `<OOV>` token : `<UNK>` un known or `Out of Vocabulary`

- Model:  
  - output softmax
  - 前瞻模型， 看到之前所有的序列。
  - That is: 
    - Each time when we try to predict the probability at a position $\hat{y}^{<t>}$, 
    - we get the activation of the former position $a^{<t-1>}$ 
    - and also the **`real value`** of the former position $y^{<t-1>}$
  ![](W1_002.png)

### 4.2 Sampling novel sequences
1. Randomly sample according to the `softmax` distribution
2. Take the `predicted output` $\hat{y}^{<t>}$ instead of the `real value` ${y}^{<t>}$ as the input of the next position until the end of the sentence. (`<EOS>`)

    The same time we `refuse` all the sample that contains `<OOV>`  
3. So that we do the prediction of the full sentence as a `sample of predicion` based on our former predicions

Word-level $\not =$ Caracter-level rnn

## 5. Problems with RNN

**Vaninishing gradients!**


**Raison:** fail to deel with long range depencency

### 5.1 Exploding gradients with RNN

- Exploding gradients : NaN   
- Solution: `Gradient clippling`  
That means, look at the gradient vectors, if it's `too big` then we make a `rescedule` so that we can continue to propogate. There are `clips` according to some maximum value.  
This is a relatively robust solution to the problem of `exploding gradients`

### 5.2 Vaninshing gradients with RNN
Long term RNN model to envite Vanishing gradients 

We will use `Gated Recurrent Unit` (GRU)  

Which is introduced by [Cho et al., 2014](https://arxiv.org/pdf/1409.1259.pdf) and [Junyoung Chung et al., 2014](https://arxiv.org/pdf/1412.3555)

memory cell c^T

Use some sigmoid-like Gama function $\Gamma_u$ to denote the gate function which decide wether we will update our memory $c^{<t>}$
$$
\begin{aligned}
  \tilde{c}^{<t>} &= \tanh (\Gamma_r * W_c {\mathop{}^{C^{<t-1>}}_{X^{<t>}}} + b_c)\\ 
  c^{<t>} &= \Gamma_u * \tilde{c}^{<t>} + (1-\Gamma_u) * c^{<t-1>}\\
  a^{<t>} &= c^{<t>}
\end{aligned}
$$

where,
$$
  \left\{ 
    \begin{array}{ll}
  \Gamma_u = \sigma(W_u {\mathop{}^{C^{<t-1>}}_{X^{<t>}}} + b_u)\\

  \Gamma_r = \sigma(W_r {\mathop{}^{C^{<t-1>}}_{X^{<t>}}} + b_r)
    \end{array}
  \right.\\
$$

## 6. Long Short Term Memory (LSTM)

Introduced by [Hochreiter & Schmidhuber 1997](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf)

Difference between GRN and LSTM:  
- Instead of using gate function  
  - update gamma : $\Gamma_u$
  - relevent gamma : $\Gamma_r$  
  
- We introduce in LSTM:
  - update gamma : $\Gamma_u$
  - `forget` gamma : $\Gamma_u$
  - output gate : $\Gamma_o$
- Then we deduce $c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + \Gamma_f * c^{<t-1>}$  
  Instead of $c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + (1-\Gamma_u) * c^{<t-1>}$
- Finally we deduce the activation with the output gate:
  $$a^{<t>} = \Gamma_o * \tanh (c^{<t>})$$  
  Instead of 
  $$a^{<t>} = c^{<t>}$$



Let's look at it in picture: ![](w1_003.png)
  
*peephole connection*: Use also $c^{<t-1>}$ as the input of 
$$\Gamma_{u,f,o}$$

`GRU` use `smaller` model, and is usually much `easier` to train, but since is experically proved with high stability, we often try `LSTM` first as the default model for this kind of `long range memory problems`.

<!-- GRN|LSTM
---|---
$\s$ |sf -->

## 7. Introduction to some other mentioned RNNs

As we were [seen](#32-problem-a-class--%22ancho%22-id--%2232%22a), traditionnal RNNs take only one `forward direction` of positions to `update`. That works generally well `may` still meets some problems because in `natural language-like text`, later relations make sense also for the meaning.

### 7.1 Bidirectional RNN
We can get the infomation from the future!

In fact, we read a text of listen to some textse, by considering the context, that is, the text `before` and `after`.

Add in the network some backward activation  
Acyclic graph :![](w1_004.png) 
So that we have both `forward activation` and `backward activation`: 
-  we begin by going forward through the whole sentence with these traditional `forward activations` 
-  and then go back through all of the `backward activations`.  
-  In this way, we can then do the prediction at each position with both `forward info` and `backward info` if there existe.  
- It  `works` for RNN and also for GRU and LSTM

**disadvantage**: we need to konw the entire sentence before we do the text recognition.

### 7.2 Deep RNNs
We can use `multiple layers` of LSTM or RNN or GRU to contruct a deep RNN network,![](w1_005.png)   
**BUT**, Since a single RNN contains already many parameters, usually 3 layers is the maximum for the network structure.