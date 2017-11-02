Name Entity Recognition with DNN

1.Structure
---------------------------------------------------------------------------------------------------------------
We use a window(default size is 3) to capture contextual information. given a sequence of tokens, we will use a fixed window size
go from left to right to generate a new sequence, the length of each sub-sequence of this new sequence is same as window size.

1).embedding for x,additional features and y; 2). first layer of FC; 3).second layer of FC; 4).softmax

2.Usage
---------------------------------------------------------------------------------------------------------------
It does not depend on any other dataset or files. Run the mode by using follow command:

python dnn_model.py

3.Toy task: test function for training and prediction
---------------------------------------------------------------------------------------------------------------
We use a sequence of indexex to represent tokens of language. Each index is associate with target label based on specific rules. 
Given a sequence of tokens, a new sequence will be generated, and will be splitted into a sequence of sub-sequence, then will
be feed to our model.
