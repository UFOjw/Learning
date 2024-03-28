# Decision-Tree-Regressor

The decision tree algorithm is a popular machine learning algorithm used for classification and regression tasks.
It is a supervised learning algorithm that can be used for both binary and multi-class classification problems.
The algorithm works by recursively splitting the dataset based on feature values to create subgroups.

MSE and Weighted MSE are used as a criterion for splitting a node.

![equation](https://latex.codecogs.com/svg.image?&space;MSE_{weghted}&space;=&space;\frac{MSE_{left}&space;*&space;N_{left}&space;&plus;&space;MSE_{right}&space;*&space;N_{right}}{N_{left}&space;&plus;&space;N_{right}})

And as a stopping criterion, two are used:
* `max_depth` (maximum depth of the tree) - if the maximum depth is reached, the node is not further divided and remains a leaf.
* `min_samples_split` (minimum number of objects in a node for further division) - if there are fewer objects, the node is not further divided and remains a leaf.

This implementation uses a depth-tree construction method. His idea is that each node divides independently until one of the stopping criteria is reached.

In order to train a tree, the class provides a `fit` method that takes 2 parameters `(X_train, y_train)`.

It also provides visualization of the tree using the `as_json` function, which turns the DecisionTreeRegressor object into a **JSON** string.

To predict results on new data, the `DecisionTreeRegressor` class has a `predict` method that takes one parameter `(X_test)`.
