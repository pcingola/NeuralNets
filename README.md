
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
- Tip: Expand 'data/timestamp' into many columns
- r^2 is a number below 1.0, it can be negative infinity (not between 0 and 1 as many people think)

### Random Forests

Ref: [Fast.ai: Introduction to random forests](http://course18.fast.ai/lessonsml1/lesson1.html)
- See Jupiter notebook `ml/randomForest/Bulldozers.ipynb`
- Random forest are easy, generic and require no prior assumptions
- To use a random forest you need to convert categorical data to numeric.
- Try sorting accordingly if the categorical data has an ordering, e.g. `['low' 'medium' 'high']` it sometimes provides a small advantage

