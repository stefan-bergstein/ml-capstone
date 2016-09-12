# Smart Performance Prediction
## Udacity Machine Learning Engineer Nanodegree - Capstone project

This project aims to automate and simplify the prediction of application bottlenecks and system overloads by applying machine learning. It is a simplified approach because the solution does not need performance metrics from all application technology layers. Monitoring and analyzing basic systems metrics should be sufficient to predict application performance and resources problems.
A reasonable analogy is a simple health check at a medical doctor that possibly will indicate various diseases by just measuring human temperate, blood pressure and heart beat.

The picture below shows the end to end approach for this solution. Multiple simulations are performed on real application servers and the solution monitors only base operating systems metrics. Supervised machine learning is used to create a model for predicting situations in production environments later on. The input data for the learning are the simulations and monitored systems metrics.

Instead of monitoring hundreds of metrics for all technology layers and requiring deep domain expert knowledge for alerting rules for, the solution learns categories of performance and resources issues.

![alt text](https://github.com/stefan-bergstein/ml-capstone/blob/master/Picture2.png "end to end approach")


Here a link to the [Project Report.](../blob/master/MLNDCapstoneProjectReport-StefanBergstein.pdf)

