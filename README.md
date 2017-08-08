This is my attempt at creating a Question Answering model for the [SQuAD database](https://rajpurkar.github.io/SQuAD-explorer/).
 
Most of the code, except from the actual model, is starter code from assignment 4 of the Stanford Course [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/). 
The actual model is created, trained, and evaluated on the validation set in code/train.py. 

Right now, there is a simple baseline model and a DCN (Dynamic Coattention Network) model, which is work in progress.
The best result so far is 25% EM (exact match) and 42% F1 on the validation set with the work in progress DCN model.
