
# Neural networks & Deep-learning

### Neural nets

- Ising model 2D
- Kohonen: A sefl-organizing neural net that leans input distribution (2D)
- 8 Rocks problem: Place 8 rocks on a chess board. Solved using continuous Ising model (mean field theory)
- 8 Queens problem: Place 8 Queens on a chess board. Using a continuous Ising model (mean field theory)

### Tensorflow

Basic sample code:

- XOR: Backpropagation with Mean Squared Error (MSE)
- XOR: Backpropagation with Average Cross-Entropy (ACE)
- MNist: Learn to categorize digits

More advanced:
- Facial recognition
- Sex identification (from facial recognition)

# Notes & Learning points

Some learning points about ML & AI.
These are in no particular order

### General ML

- Course of dimentionality is not a real concern in practice for many ML problems
- The "No free lunch" theorem doesn't usually apply in practice because practical ML problems are not totally random data (i.e. the data lies within a sub-manifold)
- `r^2 = 1-SS_reg / SS_tot`, the second term is the ratio between `SS_reg` (sum of squared errors of our model) and `SS_tot` (sum of squared error of a very naive estimator, that is the just the mean). This is a number below 1.0, it can be negative infinity (not between 0 and 1 as many people think)

### Data pre-processing

- Try sorting accordingly if the categorical data has an ordering, e.g. `['low' 'medium' 'high']` it sometimes provides a small advantage (by default most packages just sort alphabetically)
- Expand 'data/timestamp' into many columns (`year, month, day, day_of_weeK, is_holliday`, etc.)
- Saving the data in `feather` format is udualy fast (the dataFrame is just dumped from memory to disk, so it's fast)
- Dividing 80% training, 10% testing and 10% validation works usualy OK, but if you have millions of samples, you might only need 1% for test and validation (or even 0.1%)

### Exploratory data analaysis

- Calculate ranking variuables having missing values
- Create a shallow RandomForest model having a single tree: it's a crappy model, but you can draw the tree to gain insight into the data.

### Random Forests

Ref: [Fast.ai: Introduction to random forests](http://course18.fast.ai/lessonsml1/lesson1.html)
- See Jupiter notebook `ml/randomForest/Bulldozers.ipynb`
- Random forest are easy, generic and require no prior assumptions
- To use a random forest you need to convert categorical data to numeric.

Ref: [Fast.ai: Random forest deep dive](http://course18.fast.ai/lessonsml1/lesson2.html)
- Create a shallow RandomForest model having a single tree: it's a crappy model, but you can draw the tree to gain insight into the data.
- Building a tree:
  - How to build a RandomForest: Just build a bunch of Trees
  - How to build a Tree: For each variable, for each possible splits (middle point betwee in your dataset), pick the 'best' split
  - How to compare 'splits': For each split compare the r^2 of the parent node vs the weighted average of the r^2 of the child nodes


