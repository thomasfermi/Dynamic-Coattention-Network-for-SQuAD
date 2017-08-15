This is my tensor flow implementation of the [Dynamic Coattention Network](https://arxiv.org/abs/1611.01604) Question Answering model for the [SQuAD database](https://rajpurkar.github.io/SQuAD-explorer/) with tensorflow (tested with version 1.1 and 1.2).
 
The code that is not the actual model, is starter code from assignment 4 of the Stanford Course [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/). The files [abstract\_model.py](code/abstract_model.py), [simple\_baseline\_model.py](code/simple_baseline_model.py) and [DCN\_model.py](code/DCN_model.py) in the code directory were developed by me. 
Training can be started with [code/train.py](code/train.py).

To implement the model I had to learn some tensorfow functions like tf.einsum, tf.gather_nd and tf.map_fn. I did my experiments with these functions on toy data in [this notebook](Experimentation_Notebooks/toy_data_examples_for_tile_map_fn_gather_nd_etc.ipynb) in the Experimentation\_Notebooks folder.


The best result so far is 41% EM (exact match) and 59% F1 score on the validation set with the DCN_model and training was  started via
```bash
python code/train.py --batch_size=64 --rnn_state_size=150 --figure_directory=fig_HMN_3steps_drop_07/ --dropout=0.7
```