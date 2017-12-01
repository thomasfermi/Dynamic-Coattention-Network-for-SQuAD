This is my tensorflow implementation of the [Dynamic Coattention Network](https://arxiv.org/abs/1611.01604) applied to question answering for the [SQuAD database](https://rajpurkar.github.io/SQuAD-explorer/) (tested with tensorflow version 1.1 and 1.2). The network gets a Wikipedia article and a question as inputs and should predict a segment (or span) of the article that answers the question.
 
The data in the data/squad folder was downloaded and preprocessed via the starter code from assignment 4 of the Stanford Course [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/).

If you just want to have a look at the DCN implementation check out [DCN\_model.py](code/DCN_model.py), it is only around 200 lines long.

To implement the model I had to explore some tensorflow functions like tf.gather\_nd and tf.map\_fn. I did my experiments with these functions on toy data in [this notebook](Experimentation_Notebooks/toy_data_examples_for_tile_map_fn_gather_nd_etc.ipynb) in the Experimentation\_Notebooks folder.

The best result so far is 48% EM (exact match) and 64% F1 score on the validation set. Training was started via
```bash
python code/train.py --rnn_state_size=150
```

Note:

- You will need the [tqdm package](https://pypi.python.org/pypi/tqdm) to run the code
- Right now the project is on ice, due to the high costs for training on AWS instances. I might continue the project once I get a proper graphics card.

TODO:

- The hyperparameter search is not finished (e.g.: How much can using 300 dimensional word vectors improve performance compared to 100 dimensional word vectors?)
- Check influence of LSTM vs GRU
