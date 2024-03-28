# UCB

In the world there are a large number of tools for promoting services, goods, services. One of them is advertising. It is sometimes difficult and cost-effective for an advertiser to choose an audience to maximize profits.

Therefore, in the modern world, affiliate networks have appeared that provide advertisers with a place for their offer and access to hundreds of thousands of webmasters. Arbitrageurs, in turn, have statistics on offers and can choose them according to their interests on their sites.

How to understand all these questions for a beginner affiliate manager if he already has traffic, but has no experience yet?

This repository provides a mini-service that recommends an offer based on user clicks.

The service receives a `click` and a `list of offers` that we can offer. Using the `epsilon-greedy` algorithm and the `UCB` algorithm, we recommend ads to the user.

Greedy implementation:

![eps](https://storage.yandexcloud.net/klms-public/production/learning-content/55/1230/16303/47037/253254/Новый%20проект.png)

UCB implementation:

![UCB](https://latex.codecogs.com/svg.image?UCB(a)=RPC(a)&plus;c\sqrt{\frac{log(t)}{NumberOfShows(a)_{t-1}}})

We receive an answer from the service: did the user perform the target action in the format `click`, `reward`.

Also on the service you can see the statistics for each offer:
* Number of shows
* Number of targeted actions
* Conversion
* Total reward
* Conversion into action
* Reward per click
