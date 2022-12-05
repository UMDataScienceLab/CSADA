CSADA

This repository stores the code for the paper _Rethinking Cost-sensitive Classification in Deep Learning via Adversarial Data Augmentation_ (available at https://arxiv.org/abs/2208.11739). This paper proposed a method that makes deep neural networks cost-sensitive by generating directed adversarial examples.

Experiments were done on MNIST, CIFAR-10, and our medical image dataset (labeled NLM in the repository). Experimental results may be different depending on your cost structure and pre-trained models. In extreme cases where all critical pairs have low costs, the baseline model is already cost-sensitive. If you want to replicate the exact result of the paper, cost structures and pre-trained models used in the paper are stored in https://drive.google.com/file/d/19cMQjDOhZKrql7Vsmq0UPAMxvFgI12Xp/view?usp=sharing since GitHub does not allow files larger than 25MB.
