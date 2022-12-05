CSADA

This repository stores the code for the paper _Rethinking Cost-sensitive Classification in Deep Learning via Adversarial Data Augmentation_ (available at https://arxiv.org/abs/2208.11739). This paper proposed a method that makes deep neural networks cost-sensitive by generating directed adversarial examples (see figure below).

![image](https://user-images.githubusercontent.com/65100313/205723574-2a794edd-0e60-4af9-ac5f-b848c3bbd8bc.png)

Experiments were done on MNIST, CIFAR-10, and our medical image dataset (labeled NLM in the repository). 

The pre-trained baselines we use to train cost-sensitive deep networks can be found at https://drive.google.com/file/d/19cMQjDOhZKrql7Vsmq0UPAMxvFgI12Xp/view?usp=sharing (GitHub does not allow files larger than 25MB). Recall downloading the pretrained baseline files to replicate our exact results, as experimental results depend on the specified cost structures and baseline models. 

To replicate the result, follow the steps:
1) download desired folder from https://drive.google.com/file/d/19cMQjDOhZKrql7Vsmq0UPAMxvFgI12Xp/view?usp=sharing
2) file mnist_baseline_train produces a pre-trained baseline model, which is already provided in the directory. File ...\csada.py or ...\AdvAug.py will produce the result of our algorithm, ...\_penalty.py will produce results for the penalty method, and ...\_sosr.py will produce results for the smooth one-sided regression (SOSR). ...\_compare.py compares the performance of CSADA and penalty method when there is only one critical error. Please refer to our paper for more details.  
3) All the files are in place, so just run the corresponding files and get the results. 
