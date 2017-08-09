This is my attempt at creating a Question Answering model for the [SQuAD database](https://rajpurkar.github.io/SQuAD-explorer/).
 
Most of the code, except from the actual model, is starter code from assignment 4 of the Stanford Course [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/). abstract\_model.py, simple\_baseline\_model.py and DCN\_model.py in the code/ directory were developed by me. 
Training can be started with code/train.py. 

Right now, there is a simple baseline model and a DCN (Dynamic Coattention Network) model, which is work in progress.
The best result so far is 40% EM (exact match) and 57% F1 score on the validation set with the work in progress DCN model (using modified dynamic pointer decoding).
