# multivariateClassification
Multivariate classification in C# (ML model)

Process details:

1. Loading training data and initialization
2. Calculating probability of each class
3. Calculating mean of each class
4. Calculating variance of each class
5. Calculating covariance of each class
6. Clean data - remove outliers using Mahalanobis distance
7. Load testing data
8. Calculating determinants (gi) of each class
9. Selecting corresponding class for each testing data according to the max gi

Discriminant functions for every class:

Since I used C#, the matrix calculation is a little bit more complicated. I used an Accord.Math library for matrix operations, however the code for each discriminant is very long since I have to call the corresponding method of the Accord class for every single matrix calculation. For this reason, I created a function called “Discriminant” and divided the problem into 3 parts according to the discriminant formula (see picture below). The function is being called once for each gi.

![image](https://github.com/Guestik/multivariateClassification/assets/18994179/bff8c779-8f61-41d5-8321-e596e275ad1c)
