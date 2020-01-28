---
layout: post
title:  "BERT, ELMo, & GPT-2: How contextual are contextualized word representations?"
date:   2020-02-14
author: Kawin Ethayarajh 
paper-link: https://www.aclweb.org/anthology/D19-1006.pdf
link-text: (see paper)
categories: NLP
---
Incorporating context into word embeddings - as exemplified by [BERT](https://arxiv.org/abs/1810.04805), [ELMo](https://arxiv.org/abs/1802.05365), and [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - has proven to be a watershed idea in NLP. Replacing *static vectors* (e.g., word2vec) with **contextualized word representations** has yielded [significant improvements](https://gluebenchmark.com/leaderboard) on virtually every NLP task.

But just *how contextual* are these contextualized representations?

Consider the word 'mouse'. It has multiple word senses, one referring to a rodent and another to a device. Does BERT effectively create one representation of 'mouse' per word sense? Or does BERT create infinitely many representations of 'mouse', each highly specific to its context? Does the level of context-specificity differ across layers of BERT?
<p align="center">
	<img src="{{ site.url }}/blog/assets/contextual_mouse_transparent_1.png" style="width: 46%">
	&nbsp; vs. &nbsp;
	<img src="{{ site.url }}/blog/assets/contextual_mouse_transparent_2.png" style="width: 46%">
</p>

In our EMNLP 2019 paper, ["How Contextual are Contextualized Word Representations?"](https://www.aclweb.org/anthology/D19-1006.pdf), we tackle these questions and arrive at some surprising conclusions:

1. In all layers of BERT, ELMo, and GPT-2, contextualized word representations are *anisotropic*: they occupy a narrow cone in the embedding space instead of being distributed throughout.

2. In all three models, upper layers produce more context-specific representations than lower layers. However, intra-sentence similarity varies across models, suggesting that BERT, ELMo, and GPT-2 contextualize words very differently from one another.

3. On average, less than 5% of the variance in a word’s contextualized representations can be explained by their first principal component[^1]. Still, principal components of contextualized representations in lower layers of BERT outperform GloVe and FastText on static embedding benchmarks like solving word analogies![^2]


### Measures of Contextuality

What does contextuality look like? Consider these two sentences:

> <span style="font-style: normal; letter-spacing: 0px; color: black"> A panda <span style="font-style: normal; letter-spacing: 0px; color: red">dog</span> runs.</span>

> <span style="font-style: normal; letter-spacing: 0px; color: black">A <span style="font-style: normal; letter-spacing: 0px; color: green">dog</span> is trying to get bacon off its back.</span>

<span style="font-style: normal; letter-spacing: 0px; color: red">$\vec{dog}$</span> == <span style="font-style: normal; letter-spacing: 0px; color: green">$\vec{dog}$</span> implies that there is no contextualization (i.e., what we'd get with word2vec). 
<span style="font-style: normal; letter-spacing: 0px; color: red">$\vec{dog}$</span> != <span style="font-style: normal; letter-spacing: 0px; color: green">$\vec{dog}$</span> implies that there is *some* contextualization. The difficulty lies in quantifying the extent to which this occurs. Since there is no definitive measure of contextuality, we propose three new ones:

1. **Self-Similarity (SelfSim)**: The average cosine similarity of a word with itself across all contexts, where representations of the word are drawn from the same layer of a given model. For example, we would take the mean of cos(<span style="font-style: normal; letter-spacing: 0px; color: red">$\vec{dog}$</span>, <span style="font-style: normal; letter-spacing: 0px; color: green">$\vec{dog}$</span>) over all unique pairs to calculate SelfSim('dog').

2. **Intra-Sentence Similarity (IntraSim)**: The average cosine similarity between a word and its context, where the context is represented as the average of its word representations. For example, for the first sentence $\vec{s} = \frac{1}{4}(\vec{A} + \vec{panda} + \vec{dog} + \vec{runs}).$ IntraSim(s) would then be the average cosine similarity between $s$ and each of the four words.

3. **Maximum Explainable Variance (MEV)**: The proportion of variance explained by the first principal component of a word’s representations (in a given layer) across different contexts. For example, MEV('dog') would be the proportion of variance explained by the first principal component of <span style="font-style: normal; letter-spacing: 0px; color: red">$\vec{dog}$</span>, <span style="font-style: normal; letter-spacing: 0px; color: green">$\vec{dog}$</span>, and every other instance of 'dog' in the data.

Note that each of these measures is calculated for *a given layer of a given model*, since each layer has its own representation space. For example, the word 'dog' has different self-similarity values in Layer 1 of BERT and Layer 2 of BERT.


### Adjusting for Anisotropy

When discussing contextuality, it is important to consider the isotropy of embeddings (i.e., whether they're uniformly distributed in all directions).

In both figures below, SelfSim('dog') = 0.95. On the left, isotropy is high: this suggests that 'dog' is poorly contextualized, since its representations are nearly identical across all contexts. The figure on the right -- which has low isotropy -- suggests the opposite: because *any two words have a cosine similarity > 0.95*, a self-similarity of 0.95 is relatively low, in which case 'dog' *is* highly contextualized!
<p align="center">
	<img src="{{ site.url }}/blog/assets/sphere_1.png" style="width: 30%">
	&nbsp; vs. &nbsp;
	<img src="{{ site.url }}/blog/assets/sphere_2.png" style="width: 30%">
</p>
To adjust for anisotropy, we calculate *anisotropic baselines* for each of our measures and subtract each baseline from the respective raw measure.[^3]

But is it even necessary to adjust for anisotropy? Yes! As seen below, upper layers of BERT and GPT-2 are extremely anisotropic, suggesting that high anisotropy is inherent to -- or at least a consequence of -- the process of contextualization: 

<p align="center">
	<img src="{{ site.url }}/blog/assets/mean_cosine_similarity_across_words.png" style="width: 100%">
</p>


### Context-Specificity

**On average, contextualized representations are more context-specific in higher layers.** As seen below, the decrease in self-similarity is almost monotonic. This is analogous to how upper layers of LSTMs trained on NLP tasks learn more task-specific representations ([Liu et al., 2019](https://arxiv.org/abs/1903.08855)). GPT-2 is the most context-specific; representations in its last layer are almost maximally context-specific.

<p align="center">
	<img src="{{ site.url }}/blog/assets/self_similarity_above_expected.png" style="width: 100%">
</p>

**Stopwords such as 'the' have among the lowest self-similarity (i.e., the most context-specific representations).** This suggests that the variety of contexts a word appears in, rather than its inherent polysemy, is what drives variation in its contextualized representations.  This suggests that ELMo, BERT, and GPT-2 are not simply assigning one representation per word sense; otherwise, there would not be so much variation in the representations of words with so few word senses.

**Context-specificity manifests very differently in ELMo, BERT, and GPT-2.** As seen below, in ELMo, words in the same sentence are more similar to one  another in upper layers. In BERT, words in the same sentence are more dissimilar to one another in upper layers but are on average more similar to each other than two random words. In contrast, for GPT-2, word representations  in the same sentence are no more similar to each other than randomly sampled words. This suggests that BERT and GPT-2's contextualization are more nuanced than ELMo's, as they seem to recognize that words appearing in the same context do not necessarily have a similar meaning.

<p align="center">
	<img src="{{ site.url }}/blog/assets/mean_cosine_similarity_between_sentence_and_words.png" style="width: 100%">
</p>


### Static vs. Contextualized

**On average, less than 5% of the variance in a word’s contextualized representations can be explained by a static embedding.** This 5% threshold represents the best-case scenario, where the static embedding is the first principal component. There is no theoretical guarantee that a word vector obtained using GloVe, for example, would be similar to the static embedding that maximizes the variance explained. This suggests that BERT, ELMo, and GPT-2 are not simply assigning one embedding per word sense: otherwise, the proportion of variance explained would be much higher. 

**Principal components of contextualized representations in lower layers of BERT outperform GloVe and FastText on many static embedding benchmarks.** This method takes the previous finding to its logical conclusion: what if we created a new type of static embedding for each word by simply taking the first principal component of its contextualized representations? It turns out that this works surprisingly well. If we use representations from lower layers of BERT, these *principal component embeddings* outperform GloVe and FastText on benchmark tasks covering semantic similarity, analogy solving, and concept categorization. 
	
As seen below, for all three models, principal component embeddings created from lower layers are more effective than those created from upper layers. Those created using GPT-2 perform markedly worse than those from ELMo and BERT. Given that upper layers are much more context-specific than lower layers, and given that GPT-2’s representations are more context-specific, this suggests that principal components of less context-specific representations are more effective on these tasks.

<p align="center">
	<img src="{{ site.url }}/blog/assets/pc_static_embeddings.png" style="width: 100%">
</p>


### Conclusion

In ELMo, BERT, and GPT-2, upper layers produce more context-specific and representations than lower layers. However, these models contextualize words very differently from one another: after adjusting for anisotropy, the similarity between words in the same sentence is highest in ELMo but almost non-existent in GPT-2.

On average, less than 5% of the variance in a word's contextualized representations can be explained by a static embedding. Even in the best-case scenario, static word embeddings would thus be a poor replacement for contextualized ones. Still, contextualized representations can be used to create a more powerful type of static embedding: principal components of contextualized representations in lower layers of BERT are much better than GloVe and FastText!

If you're interested in reading more along these lines, check out Anna Rogers' [The Dark Secrets of BERT (2020)](https://text-machine-lab.github.io/blog/2020/bert-secrets/) and Lena Voita's [Evolution of Representations in the Transformer (2019)](https://lena-voita.github.io/posts/emnlp19_evolution.html). If you found this post useful, you can cite our paper as follows:

	@inproceedings{@inproceedings{ethayarajh-2019-contextual,
    title = "How Contextual are Contextualized Word Representations? Comparing the Geometry of {BERT}, {ELM}o, and {GPT}-2 Embeddings",
    author = "Ethayarajh, Kawin",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov, year = "2019", address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1006",
    doi = "10.18653/v1/D19-1006",
    pages = "55--65",
	}


##### Acknowledgements

<p class="small-text"> 
Many thanks to Anna Rogers for live-tweeting this paper during EMNLP 2019.
</p>


##### Footnotes

[^1]: This was calculated after adjusting for the effect of anisotropy.

[^2]: Some previous work ([Schluter, 2018](https://www.aclweb.org/anthology/N18-2039); [Rogers et al., 2017](https://www.aclweb.org/anthology/S17-1017)) has argued that word analogies should not be used for evaluating word embeddings, for a number of different theoretical and empirical reasons. We agree with this position. However, given the historical importance of this test, it is worth mentioning here.

[^3]: For self-similarity and intra-sentence similarity, the baseline is the average cosine similarity between randomly sampled representations (of different words) from a given layer's representation space. For MEV, the baseline is the variance explained by the first principal component of uniformly randomly sampled representations. See the paper for details.