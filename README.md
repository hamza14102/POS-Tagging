---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Hidden Markov Model


* `submitted.py`: Your homework. Edit, and then submit to <a href="https://www.gradescope.com/courses/486387">Gradescope</a>.
* `mp08_notebook.ipynb`: This is a <a href="https://anaconda.org/anaconda/jupyter">Jupyter</a> notebook to help you debug.  You can completely ignore it if you want, although you might find that it gives you useful instructions.
* `tests/test_visible.py`: This file contains about half of the <a href="https://docs.python.org/3/library/unittest.html">unit tests</a> that Gradescope will run in order to grade your homework.  If you can get a perfect score on these tests, then you should also get a perfect score on the additional hidden tests that Gradescope uses.
* `data`: This directory contains the data.
* `util.py`: This is an auxiliary program that you can use to read the data, evaluate the accuracy, etc.


### Table of Contents

1. <a href="#section1">Reading the data</a>
1. <a href="#section2">Tagset</a>
1. <a href="#section3">Taggers</a>
1. <a href="#section4">Baseline Tagger</a>
1. <a href="#section5">Viterbi: HMM Tagger</a>
1. <a href="#section6">Viterbi\_ec: Improveing HMM Tagger</a>



<a id='section1'></a>
## Reading the data
The dataset consists of thousands of sentences with ground-truth POS tags. 

The provided load_dataset function will read in the data as a nested list with the outer dimension representing each sentence and inner dimensin representing each tagged word. The following cells will help you go through the representation of the data.

The provided code converts all words to lowercase. It also adds a START and END tag for each sentence when it loads the sentence. These tags are just for standardization. They will not be considered in accuracy computation.

```python
import utils
train_set = utils.load_dataset('data/brown-training.txt')
dev_set = utils.load_dataset('data/brown-test.txt')
```

```python
print('training set has {} sentences'.format(len(train_set)))
print('dev set has {} sentences'.format(len(dev_set)))
print('The first sentence of training set has {} words'.format(len(train_set[0])))
print('The 10th word of the first sentence in the training set is "{}" with ground-truth tag "{}"'.format(train_set[0][9][0], train_set[0][9][1]))
```

```python
print('Here is an sample sentence from the training set:\n', train_set[0])
```

<a id='section2'></a>
<h2>Tagset</h2>

<p>
  The following is an example set of 16 part of speech tags.  
      This is the tagset used in the provided 
Brown corpus. <b><font color=red> But remember you should not hardcode anything regarding this tagset because we will test 
your code on two other datasets with a different tagset</font></b>.
</p>

<ul>

<li> ADJ adjective
<li> ADV adverb
<li> IN preposition
<li> PART particle (e.g. after verb, looks like a preposition)

<li> PRON pronoun
<li> NUM number
<li> CONJ conjunction
<li> UH filler, exclamation

<li> TO infinitive
<li> VERB verb
<li> MODAL modal verb
<li> DET determiner

<li> NOUN noun
<li> PERIOD end of sentence punctuation
<li> PUNCT  other punctuation
<li> X miscellaneous hard-to-classify items
</ul>


<a id='section3'></a>
<h2>Taggers</h2>

You will need to write two main types of tagging functions:

<ul>
<li> Baseline tagger
<li> Viterbi: HMM tagger
</ul>

For implementation of this MP, You may use numpy (though it's not needed). <b><font color=red>You may not use other non-standard modules (including nltk)</font></b>.

You should use the provided training data to train the parameters of your model and the test sets to test its accuracy. 

In addition, your code will be tested on two hidden datasets that are not available to you, which has different number of tags and words from the ones provided to you. So do NOT hardcode any of your important computations, such as initial probabilities, transition probabilities, emission probabilities, number or name of tags, and etc. We will inspect code for hardcoding computations/values and will penalize such implementations.


<a id='section4'></a>
<h2>Baseline Tagger</h2>

The Baseline tagger considers each word independently, ignoring previous words and tags. For each word w, it counts how many times w occurs with each tag in the training data. When processing the test data, it consistently gives w the tag that was seen most often. For unseen words, it should guess the tag that's seen the most often in training dataset.

#### For all seen word w:
$$Tag_{w}= \operatorname*{argmax}_{t \in T} (\text{# times tag t is matched to word w}) $$
#### For all unseen word w':
$$Tag_{w'}= \operatorname*{argmax}_{t \in T} (\text{# times tag t appears in the training set}) $$

A correctly working baseline tagger should get about 93.9% accuracy on the Brown corpus development set, with over 90% accuracy on multitag words and over 69% on unseen words.

```python
import submitted
import importlib
importlib.reload(submitted)
print(submitted.__doc__)
```

```python
help(submitted.baseline)
```

```python
import time
importlib.reload(submitted)
train_set = utils.load_dataset('data/brown-training.txt')
dev_set = utils.load_dataset('data/brown-test.txt')
start_time = time.time()
predicted = submitted.baseline(train_set, utils.strip_tags(dev_set))
time_spend = time.time() - start_time
accuracy, _, _ = utils.evaluate_accuracies(predicted, dev_set)
multi_tag_accuracy, unseen_words_accuracy, = utils.specialword_accuracies(train_set, predicted, dev_set)

print("time spent: {0:.4f} sec".format(time_spend))
print("accuracy: {0:.4f}".format(accuracy))
print("multi-tag accuracy: {0:.4f}".format(multi_tag_accuracy))
print("unseen word accuracy: {0:.4f}".format(unseen_words_accuracy))
```

#### <a id='section5'></a>
<h2>Viterbi: HMM Tagger</h2>
<p>
The Viterbi tagger should implement the HMM trellis (Viterbi) decoding algoirthm
as seen in lecture or Jurafsky and Martin.   That is, the probability of each
tag depends only on the previous tag, and the probability of each word depends
only on the corresponding tag. This model will need to estimate
three sets of probabilities:

<ul>
<li>  Initial probabilities (How often does each tag occur at the start of
a sentence?)
<li>  Transition probabilities (How often does tag \(t_b\)  follow tag
\(t_a\)?)
<li>  Emission probabilities (How often does tag t yield word w?)
</ul>

<p>
You can assume that all sentences will begin with a START token, whose tag is START.
<b><font color=red>So your initial probabilities will have a very restricted form, whether you choose to
handcode appropriate numbers or learn them from the data.</font></b> The initial probabilities shown
in the
textbook/texture examples will be handled by
transition probabilities from the START token to
the first real word.

<p>
It's helpful to think of your processing in five steps:

<ul>
<li> Count occurrences of tags, tag pairs, tag/word pairs.
<li> Compute smoothed probabilities
<li> Take the log of each probability
<li> Construct the trellis.   Notice that
for each tag/time pair, you must store not only
the probability of the best path but also a pointer to the
previous tag/time pair in that path.
<li> Return the best path through the trellis.
</ul>

<p>
You'll need to use smoothing to get good performance.
Make sure that your code for computing transition and emission probabilities
never returns zero.
Laplace smoothing is the method we use to smooth zero probability cases for calculating
initial probabilities, transition probabilities, and emission probabilities.
<p>
For example, to smooth the emission probabilities, consider each tag individually.
For a fixed tag T, you need to ensure that \(P_e(W|T)\) produces a non-zero number
no matter what word W you give it.
You can use Laplace smoothing (as in MP 2) to fill in a probability for
"UNKNOWN" which will be the return value for all words W that were not
seen in the training data.
For this initial implementation of Viterbi, use the same Laplace smoothing
constant \(\alpha\) for all tags.

<p>
This simple Viterbi will perform slightly worse than the baseline
code for the Brown development dataset (somewhat over 93% accuracy).
However you should notice that it's doing better on the multiple-tag
words (e.g. over 93.5%). <b><font color=red> Please make sure to follow the description to
implement your algorithm and do not try to do improvement in this part,
as it might make your code fail some of our test cases. You will be asked
to improve the algorithm in the next part.</font></b>

```python
help(submitted.viterbi)
```

```python
import time
importlib.reload(submitted)
train_set = utils.load_dataset('data/brown-training.txt')
# dev_set = utils.load_dataset('data/brown-test.txt')[:1]
dev_set = utils.load_dataset('data/brown-test.txt')
start_time = time.time()
predicted = submitted.viterbi(train_set, utils.strip_tags(dev_set))
# print(utils.strip_tags(dev_set))
# for sentence in predicted:
#     print(sentence)
time_spend = time.time() - start_time
accuracy, _, _ = utils.evaluate_accuracies(predicted, dev_set)
multi_tag_accuracy, unseen_words_accuracy, = utils.specialword_accuracies(train_set, predicted, dev_set)

print("time spent: {0:.4f} sec".format(time_spend))
print("accuracy: {0:.4f}".format(accuracy))
print("multi-tag accuracy: {0:.4f}".format(multi_tag_accuracy))
print("unseen word accuracy: {0:.4f}".format(unseen_words_accuracy))
```

<a id='section6'></a>
<h2>Viterbi_ec: Improving HMM Tagger (Optional, for Extra Credit only)</h2>
<p>
The previous Vitebi tagger fails to beat the baseline because it does very poorly on
unseen words.   It's assuming that all tags have similar probability for these
words, but we know that a new word is much more likely to have
the tag NOUN than (say) CONJ.
For this part, you'll improve your emission smoothing to match the real
probabilities for unseen words.

<p>
Words that appear zero times in the training data (out-of-vocabulary or OOV words) and words that appear once in the training 
data (<a href="https://en.wikipedia.org/wiki/Hapax_legomenon">hapax</a> words) tend to have similar parts of speech (POS).  
For this reason, instead of assuming that OOV  words are uniformly distributed across all POS, we can get a much better
estimate of their distribution by measuring the distribution of hapax words.
Extract these words from the training data and calculate the probability of
each tag on them.   When you do your Laplace smoothing of the emission probabilities
for tag T, scale the Laplace smoothing constant by P(T|hapax), i.e., the probability that tag T
occurs given that the word was hapax. Remember that Laplace smoothing acts by reducing probability 
mass for high-frequency words, and re-assigning some of that probability mass to low-frequency words. A large smoothing 
constant can end up skewing probability masses a lot, so experiment with small orders of magnitude for this hyperparameter. 
<p>
This optimized version of the Viterbi code should have a significantly
better unseen word accuracy on the Brown development dataset,
e.g. over 66.5%.  It also beat the baseline on overall accuracy (e.g.
95.5%).
You should write your optimized version of Viterbi under the viterbi_ec function in
submitted.py.
<p>
The hapax word tag probabilities may be different from one dataset to another, 
so your <tt>viterbi_ec</tt> method should compute them dynamically from its training data each
time it runs.
<p>
  Hints

<ul>
    <li> You should start with the original implementation of your viterbi algorithm in the 
        previous part. As long as you understand what you need to do, the change in implementation
        should not be substantial.
    <li> Tag 'X' rarely occurs in the dataset.  Setting a high
       value for the Laplace smoothing constant may overly smooth the emission probabilities
      and break your statistical computations.  A small value for the
      Laplace smoothing constant, e.g. 1e-5, may help.
    <li> It's not advisable to use global variables in your implementation since
      gradescope runs a number of different tests within the same python environment.
      Global values set during one test will carry over to subsequent tests.
</ul>


```python
help(submitted.viterbi_ec)
```

```python
import time
importlib.reload(submitted)
train_set = utils.load_dataset('data/brown-training.txt')
dev_set = utils.load_dataset('data/brown-test.txt')
start_time = time.time()
predicted = submitted.viterbi_ec(train_set, utils.strip_tags(dev_set))
time_spend = time.time() - start_time
accuracy, _, _ = utils.evaluate_accuracies(predicted, dev_set)
multi_tag_accuracy, unseen_words_accuracy, = utils.specialword_accuracies(train_set, predicted, dev_set)

print("time spent: {0:.4f} sec".format(time_spend))
print("accuracy: {0:.4f}".format(accuracy))
print("multi-tag accuracy: {0:.4f}".format(multi_tag_accuracy))
print("unseen word accuracy: {0:.4f}".format(unseen_words_accuracy))
```

<a id='section3'></a>


<a id='section4'></a>


<a id='grade'></a>
