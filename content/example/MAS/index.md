---
# see theme/archetype/default.md (at least for ananke)
title: Synaptic Plasticity and Sparse Neural Activity enable Life-Long Learning
#subtitle: "AI blog for all and none"
date: 2020-11-12
draft: true
featured: true
authors:
- Jens Bürger
tags:
#- computer vision
#- life-long learning
#- memory-aware synapses
#- incremental learning
#- deep learning
categories: [
"Language, Vision and Speech"
]
image:
  caption: ""
  focal_point: "Center"
  preview_only: true
---

## Simplicity vs complexity

The development and application of Artificial Neural Networks has long favoured computational efficiency and/or model simplicity over biological realism. For example, common models of neurons or synapses neglect any internal dynamics (e.g. history-dependent activation potential, weight decay) and simply implement fixed input-output mappings based on current parameter values. In addition, the majority of machine learning (ML) methods enforce a unidirectional procedure of training, testing, and deployment in the sense that, if a model encounters new information it has again to undergo training, testing, and deployment using not just the new, but also the old data.

While reduced complexity in model development led to significant improvements and advances in early classification and regression tasks, model and training algorithm simplicity also pose a barrier to the scalability of problem complexities that can be approached. Two recent advances in computer vision and lifelong learning at KU Leuven, extend (deep) neural networks with biologically plausible mechanisms at the neuron and synapse level and might help overcome such existing barriers.


## Lifelong Learning
{{< figure_caption fig="/stories/2020/LLL_principle.png" caption="Conceptual representation of life-long learning. Source: Aljundi et al. 2018">}}

Lifelong Learing (LLL; also lifelong machine learning) formulates the ambition to build machine learning models that can integrate new information while retaining already learned knowledge ([Silver, Yang and Li 2013](https://www.aaai.org/ocs/index.php/SSS/SSS13/paper/viewPaper/5802)). It considers the possibilities of updating model parameters or utilising existing features to learn new patterns or tasks without the need to fully retrain the given model on a complete set of data, old and new.
One of the main challenges arising is the question of forgetting. Considering a model with finite memory capacity, perpetually incorporating new knowledge also suggests the removal of old knowledge. While some forgetting is inevitable as new tasks have to be learned, what should be avoided is catastrophic forgetting, the broad, indiscriminate loss of previous knowledge while incrementally learning from new data.
In the field of computer vision, first steps toward LLL have been proposed, yet they fall short of what one would expect a comprehensive LLL scheme to include: a constant memory size, being problem agnostic, working with pre-trained models, leverage untrained data, and being adaptive with respect to what to learn and what to forget ([Aljundi et al. 2018](https://link.springer.com/chapter/10.1007/978-3-030-01219-9_9)).
Overall, LLL represents an increase in model complexities from static to dynamic memories, which points to the more intricate relations between neurons, synapses, neural activity regulation, and memories of differing temporal dynamics. Despite some existing work, LLL presents a fertile ground for research on a broad range of applications in computer vision and beyond.


## Memory Aware Synapses
Considering a finite memory capacity and, therefore, a finite set of tasks that can be learned without cross-task interference, a fundamental question for learning new information is, what to forget and what not? In other words, learning new information or tasks, which translates to the overwriting or updating of already trained parameters (i.e., weights; synaptic connections), requires mechanisms for determining which parameters can be updated so that important information or behaviours are not lost.
In response to this challenge, [Aljundi et al. 2018](https://link.springer.com/chapter/10.1007/978-3-030-01219-9_9) introduced Memory Aware Synapses (MAS). MAS permit the evaluation of parameter importance of a trained model, given a set of evaluation data. The novelty of this contribution is that parameter importance is determined as a measure of sensitivity. Applying small perturbations to individual parameters and observing the impacts on the overall output provides a quantification of parameter importance as well as it permits the use of unlabeled data for importance evaluation (using loss instead of sensitivity would require labeled evaluation data, which might not be available). Parameter importance, which is computed for each individual parameter, then determines the synaptic plasticity during subsequent training phases.
Subsequent learning includes the importance measures as a regularisation factor in the loss computation of the new task. While existing parameters or features (sets of parameters) can be reused for new tasks, changes to these parameters would incur an increase of the loss function, proportional to parameter importance. The following figure shows some results of MAS on a sequential learning task. A neural model is trained sequentially on separate subsets of data, with each phase trying to learn new data while preserving important parameters required for performing previous tasks without significant losses in accuracy. Compared to reference implementations, MAS exhibits the highest average classification accuracy as well as the lowest average task-specific loss in accuracy (forgetting).

#### Relation to Hebbian Learning
Calculating importance based on sensitivity rather than loss means that importance measures are obtained through the forward-pass of the applied inputs. This enables an adaptation of the global sensitivity measure to the Hebbian learning rule of "neurons that fire together, wire together" and the sensitivity of the output to a particular parameter can be approximated by the analysis of synaptically local activity correlations. The classification accuracy using the local rule, though slightly lower, are within the range of the results using a global sensitivity measure. While this local approach could lead to reduced computational requirements on standard computing hardware, it could be particularly beneficial for specialised neural hardware implementations, a point that will be briefly discussed later again.

{{< figure_caption fig="/stories/2020/local_hebbian_learning.png" width="400px" caption="Global vs. local importance determination. Source: Aljundi et al. (2018)">}}



## Representation Sparsity for more effective Use of Memory

While MAS preserve important synaptic connections throughout sequential learning phases, the approach does not explicitly consider overall memory capacity limitations and dense neural representations, as used by MAS, are generally more prone to suffer from forgetting. Sparse representations, such as in biological brains, permit to more effectively utilise a finite amount of neurons for the encoding of diverse and complex information ([Olshausen and Field 2004](https://www.sciencedirect.com/science/article/abs/pii/S0959438804001035)). In the context of LLL and in combination with MAS, sparsity was recently explored for its ability to more selectively develop a finite memory and thus preserve capacity for sequential learning tasks ([Aljundi, Rohrbach and Tuytelaars 2019](https://openreview.net/forum?id=Bkxbrn0cYX)).

{{< figure_caption fig="/stories/2020/representation_sparsity.png" width="400px" caption="Representation sparsity for two distinct tasks. Neural activations for task T1 in red and for Task T2 in green. Adapted from Figure 1 in Aljundi, Rohrbach and Tuytelaars (2019)">}}



Naming their approach Sparse coding through Local Neural Inhibition and Discounting (SLNID), the authors introduced a regularisation technique that enforces sparse and decorrelated neuronal activity. In connection with MAS, this facilitates efficient memory utilisation and state of the art memory loss minimisation in sequential learning tasks. SLNID leverages the following properties.
1. As already indicated, sparse neural activity is a key feature for resource-efficient encoding of complex features. Moreover, sparse coding of neural activity makes the network outputs overall more resilient toward noisy data.
2. Lateral inhibitions between neurons of the same layer penalise the simultaneous activity of neurons. Sparsity and inhibition are implemented through regularization adding a loss term, proportional to the amount of simultaneous neural activity, to the parameter optimiser (SGD).
3. As this regularisation would converge toward only single neurons representing input information, a localised inhibition is implemented to relax the regularisation effect and allow for more complex representations. Precisely, a Gaussian function, based on neuron indices, determines the distance between neurons and weighs inhibition accordingly. Thus, neuron-pairs with larger distances are allowed to exhibit simultaneous activity without incurring a penalty.
4. Under considerations of sparse neural activity and individual neurons representing distinct features, avoiding to forget in subsequent tasks is not only a question of protecting parameters, but also a question of protecting the responses of particular neurons. The authors therefore introduce the concept of neuron importance, akin to the concept of parameter importance as developed in MAS. Neuron importance is added to the regulariser to penalise any updates to behaviour of neurons important to previously learned tasks.


## Potential implications
The presented methods demonstrate important advances in computer vision in particular and for similar approaches to LLL in robotics, machine learning, or meta-learning, among others, in general. Furthermore, it might have potential implications at the intersection with two other fields of research.

#### Neuroscience
Coming back to the initial argument that ML algorithms did not prioritise neuroscientific realism, we can now observe a trend which brings computer vision/machine learning/deep learning and neuroscience closer together. MAS motivated the importance measure of synaptic connections to avoid overwriting synaptic connections when learning new information. Synaptic plasticity was lowered for important synapses and increased for synapses not relevant to previous task performance. In neuroscience, a similar idea is the transition between highly plastic short-term and barely plastic long-term memory. By introducing memory hierarchies ([Fusi, Drew and Abbott 2005](https://www.sciencedirect.com/science/article/pii/S0896627305001170), [Benna and Fusi 2016](https://www.nature.com/articles/nn.4401)) one can implement distinct subsystems for processing new or already known information. In other words, memory hierarchies might allow to maintain a fixed capacity for processing incoming information (short-term memory) and only transfers recurring patterns to the long-term memory, thus being selective in filling the long-term memory (high importance parameters).  

Learning features from natural images was also studied widely in neuroscience with the aim of finding biologically plausible coding principles. Two principles proved to be fundamental to this: sparse coding and synaptically local updating rules. [Olshausen and Field (1996)](https://www.nature.com/articles/381607a0) have shown that biologically realistic visual features can be learned through the regulation of sparsity in neural activation. Later, it was shown that model-wide activation sparsity can be achieved through dynamic modulation of neuron-specific activation thresholds that control a neuron's sensitivity to subsequent stimuli as a function of its activation history ([Zylberberg, Murphy and DeWeese 2011](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002250)).
These insights correlate closely with the intention of MAS and Selfless Sequential Learning. However, research related to mixed selectivity, which could be taken as an alternative interpretation of sequential learning, also shows that sparseness modulates a trade-off between discrimination and generalisation ([Barak, Rigotti and Fusi 2013](https://www.jneurosci.org/content/33/9/3844)) and that mixed selectivity offers computational advantages for complex input-output functions ([Rigotti et al. 2013](https://www.nature.com/articles/nature12160)). While there remain important differences across common neural network models and models used in computational neuroscience - mainly a question concerning the realism and complexity of neurons and synapses vs scalability - modulating synaptic plasticity and sparseness in (deep) neural networks could hold significant further potential for future research, especially when integrating insights from both fields.

#### Neuromorphic Systems

Considerable electrical energy consumption is one of the side effects of training large machine learning models in general and deep neural networks in particular ([Ho 2019](https://www.technologyreview.com/2019/06/06/239031/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/)). This stems from the fundamental mismatch between hardware and algorithmic structures. Building specialised neural hardware can reduce energy consumption as well as training times by orders of magnitude. While there already exists commercially available neural hardware, often they only support inference and training has to be done off-line (i.e., [Google's Coral edge TPU](https://coral.ai/products/accelerator/)). Learning algorithms that do not require the back-propagation of some global value (e.g. loss, error) lend themselves much better to physically realisable structures and are a key to neural hardware that supports both, training and inference. The discussed work on synaptic plasticity and sparseness potentially present important features for the future implementation of fast and energy efficient neural hardware architectures.
