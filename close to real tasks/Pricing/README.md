# Pricing

To find the price in pricing problems, one often pays attention to similar products. To compare the similarity, you need to present them in one form - in the form of embeddings.

We will assume that they are received and, in addition, we have prices for these goods.

```
embeddings = {
    1: np.array([-14.88, -2.28, 8.1, -0.11, 74.8, ..., -82.78])
    2: np.array([-50.18, 42.27, 86.07, 18.71, -18.66, ..., -78.95])
    3: np.array([-27.97, 24.00, 56.65, 3.51, 95.57, ..., -36.68])
    4: np.array([-3.0, -42.59, -2.3, 73.36, 29.98, ..., 12.43])
    5: np.array([92.91, 4.19, 5.42, 10.11, 98.34, ..., 9.10])
    ...
}


prices = {
    1: 80.5,
    2: 10.2,
    3: 55.0,
    4: 12.1,
    5: 211.2
    ...
}
```

The static methods of the `SimilarItems` class solve this problem:
* `similarity` - calculates pairwise similarities between all embeddings, returning a dictionary of similarities.
Input:
- product embeddings.
Output:
- dictionary of similarities.
* `knn` - displays a list of nearby products.
Input:
- `similarity` output.
Output:
- a dictionary with `item_id` pairs - a list of `top` nearest items.
* `knn_price` - returns the weighted average price of nearest neighbors.
Input:
- `knn` output.
Output:
- weighted average price.
