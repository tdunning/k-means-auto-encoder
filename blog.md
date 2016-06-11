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


