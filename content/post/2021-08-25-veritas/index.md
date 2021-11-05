---
title: "Versatile Verification of Tree Ensembles using Veritas"
authors: ["Laurens Devos", "Wannes Meert", "Jesse Davis"]
date: 2021-08-10
tags:
- verification
categories: ["Machine Learning & Data Science"]
social:
- icon: envelope
  icon_pack: fas
  link: 'about/#contact'  # For a direct email link, use "mailto:test@example.org".
featured: true
image:
  placement: 1
  caption: ""
  focal_point: "Center"
  preview_only: true
share: true
summary: "Evaluating machine learning models has mainly relied on reporting performance metrics. It is now becoming more common for deployed machine learned models to have to conform to requirements (e.g., legal) or exhibit specific properties (e.g., fairness). Verification techniques want to find one or multiple outcomes that match a property or to prove that such a property can never occur. We have developed two different approaches to enable verification of tree-based models."
draft: true
---

{{% callout note %}}
This content is reposted with permission from [DTAI Stories](https://dtai.cs.kuleuven.be/stories/post/laurens-devos/veritas/)
{{% /callout %}}


{{% relpub %}}
This post is based on the following publications:

- Laurens Devos, Wannes Meert, and Jesse Davis. "[Versatile Verification of Tree Ensembles.](http://proceedings.mlr.press/v139/devos21a.html)" To appear in the Proceedings of the 38th International Conference on Machine Learning. 2021.
- Laurens Devos, Wannes Meert, and Jesse Davis. "[Verifying Tree Ensembles by Reasoning About Potential Instances.](https://epubs.siam.org/doi/10.1137/1.9781611976700.51)" Proceedings of the 2021 SIAM International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2021.

Source code is available on [GitHub](https://github.com/laudv/veritas).
{{% /relpub %}}



Evaluating machine learning models has mainly relied on reporting performance metrics (e.g., accuracy, ROC AUC, squared error) on a so-called test set – a held-aside portion of the data that was not used for training the model. A good value for these metrics is usually sufficient to convince us that the model successfully learned what it needed to learn. After all, the model can predict values for examples it has never seen before.

{{< figure src="traditional-ml.png" caption="We usually split our data into a train and test set, train a model on the train set, and evaluate the model on the test set" alt="Illustrative view of traditional evaluation methods" width="80%" >}}

It is becoming more common for deployed machine learned models to have to conform to requirements (e.g., legal) or exhibit specific properties (e.g., fairness). This has motivated the development of verification approaches that are applicable to learned models. Given a specific property, these techniques verify, that is, prove whether the property holds for the model. For applications where these requirements are crucial, just achieving good predictive performance on an unseen test set is no longer sufficient for selecting a model.

{{< figure src="verifier.png" caption="A verification tool solily considers the model and verifies whether user-provided properties and patterns are present in the model" width="30%">}}

**An example:**

Assume you want to deploy a machine learning model in a hospital to predict the probability that a patient will suffer from a stroke. You train an XGBoost model on historical data and this model’s performance is convincing, nearly surpassing the capabilities of a human expert. Despite this encouraging performance, you may still want to know the answers to the following questions before deploying the model in the real world:
- Is this model robust? That is, can small, nearly imperceptible changes in some attribute values cause the model to change its prediction dramatically? A non-robust model is susceptible to adversarial examples; many fascinating examples of adversarial examples exist in computer vision, but the problem exists in nearly all machine learning algorithms.
- A related question with a different goal is: Given an example from the dataset, how can we change some of the attribute values for the individual such that the model’s predicted probability that a patient will have a stroke is as close to zero as possible. This is sometimes called a counterfactual explanation, but this must not be confused with a counterfactual in the field of causality.
- Does the model systematically under (or over) estimate the risk of a stroke for certain classes of patients (e.g., based on a patient’s race).
- Is it possible that the model predicts a probability lower than 5% for a male individual older than 60 years old with hypertension? In this case, we would need completions for the other required pieces of information (i.e., feature values) such that, when evaluated by the model, this person’s predicted probability is lower than 5%.

Unless these situations occur in the data, traditional testing will not reveal the answers to these questions. Nevertheless, the answers to these questions could be very valuable because they give insights into the model that go beyond a simple evaluation metric.

In order to answer these questions, we need to reason about all possible outcomes of a machine learning model. Specifically, we either want to find one or multiple outcomes that match a property, or we want to prove that such a property can never occur. This process is called **verification**.

We have developed two different approaches to enable verification of tree-based models. One is based on theorem proving (SDM’21), and another is based on search (ICML’21). Intuitively, the first approach answers the question “is it possible that an example with certain properties exists for which the model has a certain outcome?” The second can be understood as “what is the maximum model output for examples with certain properties?”

For example, in both papers, we look at a model which predicts view counts of YouTube videos given their titles and descriptions, and we ask the question: “Which words do we need to add to maximally change the output of the model?” We discovered that it is easy to find outputs that vary widely even though the input is mostly the same. Some examples:

- Given the words ‘live’, ‘breaking’, ‘news’, and ‘war ‘, adding words ‘big’ and ‘trailers’ increases the prediction by 2 orders of magnitude.
- Given the words ‘epic’ and ‘challenge’, adding the words ‘album’, ‘video’, and ‘remix’ increases the prediction by almost 5 orders of magnitude.

Clearly, the stakes are not high for this task. However, the same machine learning techniques could be applied to predict the validity of an insurance claim, or to detect whether an email is spam. In these cases, having the ability to carefully choose words so that the output of the model is known in advance, is a clear security issue.

A second example of verification problem is generating adversarial examples. The question here is: “Can we make imperceptible changes to a correctly classified input example so that the model’s output changes?”

![adversarials.png](adversarials.png)

The top row shows correctly classified MNIST digits. The bottom row shows slightly modified digits that are all incorrectly classified as a 9. The only exception is the last 8. The verification tool was able to prove that, for this particular digit 8, no small change exists that causes the label to flip to a 9.

We hope this short overview gives you an idea of what veification in machine learning entails. More details are available in the two papers.
