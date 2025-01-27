# KNN Classification + Prototype selection

### Problem Statement

One way to speed up nearest neighbor classification is to replace the training set by a carefully chosen
subset of “prototypes” – i.e. instead of keeping the entire training set to use for nearest neighbor classification,
carefully choose a small but representative subset to search for nearest neighbors in. Because this set is smaller
than the training data, search will be faster and thus so will classification.

### Idea

Develop a k-cluster algorithm to select k points from the training set which would act as a substitute for the whole training set.
After that we develop an 1NN classification to test the accuracy of the test data on the prototype we selected. 
