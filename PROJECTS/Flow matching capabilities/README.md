Assumption: Based on the ideas that
* By parameterizing the ODE, as done in [Neural ODE](https://arxiv.org/pdf/1806.07366), a vector field can be learned. Such a technique can more accurately determine the addition part than ResNet.
* A parameterized ODE allows one to find a way (in the case of a distribution) how a set of data (e.g. images) can be transformed into some distribution (normal) ([source](https://arxiv.org/pdf/2210.02747)).

Using the insights from Neural ODE, one can conclude that a sequence of such ODEs can replace the backbone of modern detection architectures (EfficientNet).

How it works:
A special case of the ODE solution (as described in [Flow Matching](https://arxiv.org/pdf/2210.02747)) is the diffusion model.
That is, having determined the number of steps, its intermediate ones are some features (the way in which it got from the data distribution to the normal distribution).
SD-1.5v is taken as such architecture

The experiment was conducted on [RSNA](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) data
