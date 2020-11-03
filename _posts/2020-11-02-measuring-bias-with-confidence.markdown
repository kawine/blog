---
layout: post
title:  "Measuring Bias in NLP (with Confidence!)"
date:   2020-11-02
author: Kawin Ethayarajh 
paper-link: https://www.aclweb.org/anthology/2020.acl-main.262/
link-text: (see paper)
categories: NLP
---
Countless studies have found that "bias" -- defined in manifold ways -- pervades the [embeddings](https://arxiv.org/abs/1904.03310) and [predictions](https://arxiv.org/abs/1804.09301) of the black-box models that dominate natural language processing (NLP). This, in turn, has led to a wave of work on how to "[debias](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-d)" models, only for others to find ways in which debiased models [are still biased](https://arxiv.org/abs/1903.03862), and so on.

But are these claims of NLP models being biased (or unbiased) being made with enough evidence?

Consider the sentence _"The doctor gave instructions to the nurse before she left."_ If a co-reference resolution system mistakenly thinks "she" refers to the nurse, is it gender-biased? Possibly -- but it may also make mistakes in the other direction with equal frequency (e.g., thinking "he" refers to a nurse when it doesn't). What if the system makes gender-stereotypical mistakes on not one sentence, but 100, or 1000? Then we could be more confident in claiming that it's biased.

In my ACL 2020 paper, "[Measuring Fairness under Uncertainty with Bernstein Bounds](https://www.aclweb.org/anthology/2020.acl-main.262/)", I go over how, in the haste to claim the presence or absence of bias, the inherent uncertainty in measuring bias is often overlooked in the literature:

- **Bias is not a single number**. When we test how biased a model is, we are *estimating* its bias on a sample of the data; our estimate may suggest the model is biased or unbiased when in reality the opposite is true.

- **This uncertainty can be captured using confidence intervals.** Instead of reporting a single number for bias, you should report an interval -- based on factors like your desired confidence and how you define "bias".

- **Existing datasets are too small to conclusively identify bias.** Existing datasets for measuring specific biases can only be used to make 95% confidence claims when the bias estimate is egregiously high; to catch more subtle bias, the NLP community needs bigger datasets.


### Bernstein-Bounded Unfairness

A bias estimate, made using a small sample of annotated data, likely differs from the true bias (i.e., at the population-level). How can we express our uncertainty about the estimate? We propose a method called Bernstein-bounded unfairness that translates this uncertainty into a confidence interval.

Let's say we want to measure whether some protected group $A$ is being discriminated against, relative to some unprotected group $B$. They occur in the population with frequency $\gamma_A, \gamma_B$ respectively. We need

- an annotation function $f$ that maps each example $x$ to $A, B,$ or neither

- a cost function $c : (y, \hat{y}) \to [0,C]$ that describes the cost of predicting $\hat{y}$ when the label is $y$, where $C$ is the maximum possible cost

We want to choose these functions such that our bias metric of choice -- which we call the *groupwise disparity* $\delta(f,c)$ -- can be expressed as the difference in expected cost borne by the protected and unprotected groups:

$$\delta(f,c) = \mathbb{E}_a[c(y_a, \hat{y}_a)] - \mathbb{E}_b[c(y_b, \hat{y}_b)]$$

If the protected group is incurring higher costs in expectation, it is being biased against. So what might these cost and annotation functions look like for some canonical bias metrics?

- Demographic parity requires that the probability of a positive outcome (i.e., $\text{Pr}[\hat{y} = 1]$) be equal across groups. Here, the cost $c(y, \hat{y}) = (1 - \hat{y})$.

- [Equal opportunity](https://arxiv.org/abs/1610.02413) requires that the true positive rates be equal across groups. The cost function would still be $c(y, \hat{y}) = (1 - \hat{y})$, but the annotation function would only allow qualified examples (i.e., $y(x) = 1$) into $A$ or $B$, since we're measuring the difference in true positive rates.

For a desired confidence level $\rho \in [0,1)$, a dataset of $n$ examples, and the variance $\sigma^2$ of the amortized groupwise disparity across examples, the confidence interval $t$ would be 

$$\begin{aligned}
t &= \frac{B + \sqrt{B^2 - 8 n \sigma^2 \log \left[\frac{1}{2} (1 - \rho) \right]}}{2n} \\
\text{where } B &= -\frac{2 C}{3 \gamma} \log \left[ \frac{1}{2} (1 - \rho) \right],  \gamma = \min(\gamma_A, \gamma_B)
\end{aligned}$$

For example, if we set $\rho = 0.95$, we could claim with 95% certainty that the true bias experienced by the protected group lies in the interval $[ \hat{\delta} - t, \hat{\delta} + t]$, where $\hat{\delta}$ is our bias estimate.


### Why We Need Bigger Datasets 

If we want to say with 95% confidence that a classifier is biased *to some extent* -- but want to spend as little time annotating data as possible -- we need to find the smallest $n$ such that $0 \not\in [ \hat{\delta} - t, \hat{\delta} + t]$. We can do this by working backwards from the formula for $t$ given above (see paper for details).

Let's go back to our original example. Say we have a dataset of 500 examples for measuring gender bias in co-reference resolution system, to test whether a model does better on gender-stereotypical examples (e.g., a female nurse) than non-gender-stereotypical examples (e.g., a male nurse). 

On this dataset, our bias estimate for a model we're evaluating is $\bar{\delta} = 0.05$. Is this enough to claim with 95% confidence that the model is gender-biased?

In this scenario $C = 1, \bar{\delta} = 0.05, \rho = 0.95$. We assume that there are equally many stereotypical and non-stereotypical examples and that the variance is maximal, so $\gamma = 0.5, \sigma^2 = 4$. 

With these settings, $n > 11903$; we would need a dataset of more than 11903 examples to claim with 95% confidence that the co-reference resolution system is gender-biased. This is roughly 3.8 times larger than [WinoBias](https://arxiv.org/abs/1804.06876), the largest dataset currently available for this purpose. As seen below, we could only use WinoBias if $\bar{\delta} = 0.0975$ -- that is, if the sample bias were almost twice as high.

<p align="center">
	<img src="{{ site.url }}/blog/assets/bbu_3.png" style="width: 60%">
</p>

What if we want to catch more subtle bias? Although it may be possible to derive tighter confidence intervals, what we really need are larger bias-specific datasets. Although the datasets we currently have are undoubtedly helpful, they need to be much larger in order to be a useful diagnostic.


If you found this post useful, you can cite our paper as follows:

	@inproceedings{ethayarajh-2020-classifier,
    title = "Is Your Classifier Actually Biased? Measuring Fairness under Uncertainty with Bernstein Bounds",
    author = "Ethayarajh, Kawin",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.262",
    doi = "10.18653/v1/2020.acl-main.262",
    pages = "2914--2919",
	}



##### Acknowledgements

<p class="small-text"> 
<!-- Many thanks to Krishnapriya Vishnubhotla, Jieyu Zhao, and Hila Gonen for their feedback on this blog post!  -->
</p>

