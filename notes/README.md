# Messy notes from various sites

I am using this README to drop in anything that I don't want to forget or want to keep for future reference.

### Picking an optimizer in Keras

![Different optimizers plotted with their training times and learning rates](https://cdn-images-1.medium.com/max/1200/1*gUHTqcK1PYR1EfyYAiCrmQ.png)

* Every optimizer has a different interval of learning rates where it successfully converges/trains
* There is no learning rate that works for all optimizers
* Adam is stable, learns the fastest and has a wide range of successful learning rates
* Best learning rate performances for Adam according to the article: 0.0005, 0.001, 0.00146
* Adam scales well with bigger models

Sources:
* <https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2>

### Loss functions for classification

* Cross-Entropy/Log-loss:
	* heavily penalizes predictions that are *wrong* but highly confident
	* In multiclass classification, the loss is calculated seperately for each class label per observation and the result is sumed

* Hinge Loss:
	* Intented for _binary_ classification where target values are in set {-1, 1}
	* Sometimes better than cross-entropy

* Square Hinge Loss:
	* Extension of Hinge loss
	* Easier to work with than Hinge loss from numerical point of view

* Multiclass Cross-Entropy:
	* Requires one-hot encoding (introduces significant memory usage with high number of classes)

* Sparse Multiclass Cross-Entropy:
	* Solution to the memory problem
	* No need for one-hot encoding

* Kullback Leibler Divergence loss

Sources:
* <https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23>
* <https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html>
* <https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/>