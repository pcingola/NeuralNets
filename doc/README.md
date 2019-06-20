
# ML & AI: Notes & Learning points

Some learning points about ML & AI.
These are in no particular order


# General ML

- Course of dimentionality is not a real concern in practice for many ML problems
- The "No free lunch" theorem doesn't usually apply in practice because practical ML problems are not totally random data (i.e. the data lies within a sub-manifold)
- `r^2 = 1-SS_reg / SS_tot`, the second term is the ratio between `SS_reg` (sum of squared errors of our model) and `SS_tot` (sum of squared error of a very naive estimator, that is the just the mean). This is a number below 1.0, it can be negative infinity (not between 0 and 1 as many people think)

# Data pre-processing

- Try sorting accordingly if the categorical data has an ordering, e.g. `['low' 'medium' 'high']` it sometimes provides a small advantage (by default most packages just sort alphabetically)
- Expand 'data/timestamp' into many columns (`year, month, day, day_of_weeK, is_holliday`, etc.)
- Saving the data in `feather` format is udualy fast (the dataFrame is just dumped from memory to disk, so it's fast)
- Dividing 80% training, 10% testing and 10% validation works usualy OK, but if you have millions of samples, you might only need 1% for test and validation (or even 0.1%)

# Exploratory data analaysis

- Calculate ranking variuables having missing values
- Create a shallow RandomForest model having a single tree: it's a crappy model, but you can draw the tree to gain insight into the data.
- For large datasets, there is no point on always using all the samples to train the model during the explaratory analysis: Just sub-sample your dataset

# Ensemble models
- Bagging: Create many models that are somewhat predictive, but have un-correlated errors. When you average all the models, you have `mean(prediction_i + error_i) = mean(prediction_i) + mean(error_i)`. The second term tends to zero (errors are uncorrelated with mean 0), so we have a much better predition.
- Bag if little bootstraps: It is more important to be uncorrelated than predictive (e.g. Extremelly randomized trees)


# Random Forests

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
  - In practice: Gini gain is often used
- OOB: Out of bag predictions. Use the samples that were not used by bootstrapping to test the Tree
- Tunning the RandomFores model:
	- `n_estimators`: Suggested values are `{10, 50, 100}`. Increasing the number of trees in the RandomForest reached a plateau at some point. Usualy this is after 100 trees for small datasets
	- `min_samples_leaf`: Suggested values are `{1, 3, 10, 100}`. By default RandomForest stops building the tree when every sample is a leaf node. Use `min_samples_leaf > 1` to stop earlier (you want generalization, going all the way down will overfit the tree).
	- `max_features`: Suggested values are `{0.5, 'srqt', 'log2'}`. To create uncorrelated trees, you can randomly limit the number of parameters (i.e. columns in your data frame):e.g. `max_features=0.5` other good options are `'sqrt', 'log2'`


