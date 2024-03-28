from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cosine


class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        keys = list(embeddings.keys())
        keys = sorted(keys)
        pair_sims = {}
        while len(keys) != 1:
            consEmbed = keys.pop(0)
            for otherEmbed in keys:
                similarity = (cosine(embeddings[consEmbed], embeddings[otherEmbed]) - 1) * (-1)
                pair_sims[(consEmbed, otherEmbed)] = np.round(similarity, 8)
        return pair_sims

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        itemsWithReapites = []
        for key in sim.keys():
            itemsWithReapites.extend(key)
        items = list(set(itemsWithReapites))

        knn_dict = {i: [] for i in items}
        for k, v in sim.items():
            pointer = knn_dict[k[0]]
            pointer.append((k[1], v))
            pointer = knn_dict[k[1]]
            pointer.append((k[0], v))

        for k, v in knn_dict.items():
            knn_dict[k] = sorted(v, key=lambda x: x[1], reverse=True)[:top]
        return knn_dict

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}
        for k in prices.keys():
            pricesOfTheNearest = []
            nearest = knn_dict[k]
            for item, cosineDist in nearest:
                pricesOfTheNearest.append((prices[item], cosineDist + 1))
            knn_price_dict[k] = np.round(weghtedAvg(pricesOfTheNearest), 2)
        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        pairSims = SimilarItems.similarity(embeddings)
        knnDict = SimilarItems.knn(pairSims, top)
        knn_price_dict = SimilarItems.knn_price(knnDict, prices)
        return knn_price_dict

def weghtedAvg(prices: List[Tuple[float, float]]):
    """Calculate weghted avarage"""
    numerator = 0
    denominator = 0
    for price, coef in prices:
        numerator += price * coef
        denominator += coef
    return numerator / denominator
