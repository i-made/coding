## Quickstart

### Reproducing Results

## Guidelines

  - [Quickstart](#quickstart)
    - [Reproducing results](#reproducing-results)
  - [Guidelines](#guidelines)
      - [Approach](#approach)
        - [Pros](#pros)
        - [Cons](#cons)
        - [Results and Discussion](#results-and-discussion)
      - [Performance](#performance)
        - [Time Complexity](#time-complexity)
        - [Space Complexity](#space-complexity)
      - [Future Scope](#future-scope)
        - [Scaling Up](#scaling-up)
        - [To do](#to-do)
        - [Key Element Optimisation](#key-element-optimisation)
      - [How to improve](#how-to-improve)
        - [Active learning](#active-learning)
        - [Choosing knowledge base](#choosing-knowledge-base)
        - [Links to datasets](#links-to-datasets)

### Approach

Answers to the following questions are given in the subsequent sections:

What are the pros and cons of your approach?

Which terms do you get correct, which do you get incorrect, and why?

What trade-offs (if any) did you make and why?

#### Pros
#### Cons
#### Results and Discussion

### Performance

#### Time Complexity
Time complexity in the generation of Document Term Matrix and similarity search remains O(n^2) but can be reduced further.
#### Space Complexity 
O(k) * 3 where k is the number of non zero elements in the sparse representation of the dense document term matrix.

Sparse representation of dense matrix in the LIL (list of lists) form reduces both time and space complexity of the search part. This approach has shown significant improvement in the running time for less number of entities and will certainly be helpful if there are large number of entities.

### Future Scope
Answers to the following questions are given in the subsequent sections:

What are your thoughts on future work for this project?

what considerations would you need to make for scaling up the project?

What trade-offs (if any) did you make and why?

#### Scaling Up
#### To do
#### Key Element Optimisation

### How to improve
Answers to the following questions are given in the subsequent sections:

What would you do differently with more time? 

What are the key elements that would require more thought? 

What would you like to know or learn if you worked more on this project?

#### Active learning
#### Choosing knowledge base
#### Links to datasets





