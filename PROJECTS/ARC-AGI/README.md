# Transformation of an image into actions
Idea: based on a matrix image, that is colored and has a specific shape, to extract the sequence of actions to obtain it.

Competition source: https://www.kaggle.com/competitions/arc-prize-2024

Test example ([source](https://arcprize.org/)) :

![image](https://github.com/user-attachments/assets/d8b05bd6-1093-459e-a365-850e97fdff7a)

How it works:
All pipeline is elaborated on an agent system: proposer, extractor, and validator.

The proposer initializes possible solutions (the number could vary but in the experiment = 10). In the second stage, the extractor tries to combine all possible solutions into one.
Later, another agent tries to solve a puzzle according to the text description, and if something is not satisfied then he reveals that and passes data to the next agent.
Last proposes new solutions and passes them into the extractor.

The process stops when the number of cycles reaches a maximum value (15 in the experiment).

Also, there is another process that fine-tuned the results from the previous stage. It differs in prompts and parameters.

Scripts and descriptions are provided in the [link](https://www.kaggle.com/datasets/ufo137/arc-training-description)
