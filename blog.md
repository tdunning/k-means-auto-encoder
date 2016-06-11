I got some email recently from Michael Clark about one of the anomaly detection approaches that I described in the
book [Innovations in Machine Learning: Anomaly Detection|https://www.mapr.com/practical-machine-learning-new-look-anomaly-detection]. My correspondent said:

> While I was doing my research, I came across this paper:  http://www.cs.ucr.edu/~eamonn/meaningless.pdf . I assume
> that you are aware of it, and I wonder if you think it invalidates in any way the anomaly detection approach you
> laid out in your book?

And I replied:

Yes. I have read this paper.

They assume a definition for provocatively chosen term "meanginglessness" which is itself fairly peripheral to the problem of anomaly detection. What they are saying is that k-means clustering of sliding window data is not necessarily repeatable in the sense that multiple runs of k-means on the same data does not necessarily produce cluster centroids that are nearly the same from run to run.

Unfortunately for the thesis of the Eamon's paper, this isn't an important criterion for the use of k-means in anomaly detection. A much more important criterion is that the cluster produces a model that reconstructs the data well. The fact that there are many such models that the clustering algorithm might have found that are all nearly as good is not a problem ... in fact, it can be considered a virtue since it makes it more likely that the clustering will find an acceptable solution. That is really good since k-means is unlikely to find the absolute best clustering in any case.

The reason that reconstruction quality of unseen data is the most important criterion is that the k-means clustering in this case is being used not so much in the sense of a clustering as in the sense of an auto-encoder and information bottleneck. The point is that if we can encode information in a time series in a way that uses very few bits but which still reconstructs the original signal accurately, we can say that this auto-encoder has captured important information intrinsic to the class of signals that we are looking at. Essentially, the auto-encoder has discovered *some* representation of the low-dimensional manifold that the data is constrained to occupy.

If the auto-encoder is doing this well, then it has also determined which signals might be considered plausible signals which in turn determines which signals are implausible in the sense that they cannot be reduced to the signal manifold. This can be detected by the fact that the auto-encoder cannot reproduce the original signal at all well.

I chose k-means as an algorithm for this exposition precisely because it was so very simple. IN many applications, you would need to use a more interesting representation for the auto-encoder. You might even have to use something as complex as a recurrent neural network. For my purposes, simplicity of description and implementation was more important to me, so I went with the clustering approach.

The k-means clustering approach is also used in many signal processing chains for similar reasons that I chose it ... it is simple and can be to accurately reproduce signals. These applications are also similar to mine in that none of them particularly care whether the clustering will be particularly repeatable .. only that it produce a good result.

# An Example
To illustrate the ideas here, I have created a two simple examples. These involve two kinds of randomly generated data. The first kind of data is random points that are situated on a curved 3-dimensional space that is embedded in a 15 dimensional space. These points are generated in 3 dimensions first and then a randomized neural network is used to distort and project the points into 15 dimensions. The second kind of data is constructed by constructing randomized pulses that are added to a time-series signal at randomized times. These pulses occasionally overlap making the analysis of the signal a bit harder.

In both cases, I used k-means to analyze the signal. For the random points, I simply clustered the training data and approximated each point by the nearest cluster centroid. For the time series data, I used a windowing approach like the one that I previously used in the anomaly detection book referenced above.

More details can be found, including open source code for the examples and diagrams showing the results in the [github repository|https://github.com/mapr-demos/k-means-approximation].

The quick summary is that approximation by k-means works exactly as predicted. For the 3-dimensional points data, the cube root of the number of clusters is proportional to one over the error, exactly as you would expect. This works all the way out to 2000 clusters created from 10,000 data points. With that many clusters and so few data points, the k-means algorithm is very unlikely to produce a stable clustering since many of the clusters will contain only a single data point and exactly which points form such singletons will vary from run to run. In spite of the fact that the clustering is not robust in the sense of repeatable, it is very robust in the sense of approximating previously unseen data from the same distribution.

For the time-series signal, the situation is even better. The apparent dimensionality of the problem is 32 because tha is the size of the windows that I used. Because the random pulses often don't overlap, however, the effective dimension of the data set is much lower and the error decreases very rapidly with increasing numbers of clusters. Again, the performance on previously unseen data is nearly as good as on the training data.

Of particular note is the fact that although there is obviously some overfitting going on based on the fact that the held-out test data had higher reconstruction error than the original data, increasing the number of clusters never had increased the reconstruction error on the held-out data.


