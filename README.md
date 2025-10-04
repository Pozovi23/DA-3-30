# Tool for shoulders, arms and head detection

# Get started
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
--------------------
# Some problems
According to the task, I should use `load_boston` dataset, but it was deleted due to 
ethical problems, that's wht I took `load_iris` dataset.

There was a requirement to obtain a difference greater than 0.2 between 
the Spearman and Pearson correlation, but it is not possible to get
such a big difference on real dataset. That's why I generated synthetic
non-linear data. You can see it on scatter plot in output/synthetic_plot.png
and compare to output/iris_plot.png

------------------------
# Conclusion
Spearman's correlation detects nonlinear relationships because it measures the monotonic 
association between variables using their ranks, not their actual values. Unlike Pearson's 
correlation, which assumes linearity, Spearman captures any consistent increasing or decreasing trend,
making it effective for nonlinear dependencies like exponential or quadratic relationships.# DA-3-30
