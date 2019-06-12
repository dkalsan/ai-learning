# Messy notes from various sites

I am using this README to drop in anything that I don't want to forget or want to keep for future reference.

### Picking an optimizer in Keras

[Different optimizers plotted with their training times and learning rates](https://cdn-images-1.medium.com/max/1200/1*gUHTqcK1PYR1EfyYAiCrmQ.png)

* Every optimizer has a different interval of learning rates where it successfully converges/trains
* There is no learning rate that works for all optimizers
* Adam is stable, learns the fastest and has a wide range of successful learning rates
* Best learning rate performances for Adam according to the article: 0.0005, 0.001, 0.00146
* Adam scales well with bigger models

Sources:
* <https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2>
