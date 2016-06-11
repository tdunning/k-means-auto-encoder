### fixed seed to allow repeatable results
set.seed(63)

### Original driver data. This will force our final results to be constrained
### to a 3-dimensional manifold
dim = 4
hidden = 10
embed = 15
raw.train = matrix(rnorm(dim*10000), ncol=dim, nrow=10000)
raw.test = matrix(rnorm(dim*10000), ncol=dim)

### Transformation to final data. This is a simple neural network kind of thing.
w1 = matrix(rnorm(dim*hidden), nrow=dim)
w2 = matrix(rnorm(hidden*embed), nrow=hidden)

sig = function(x) {1/(1+exp(-x)) - 0.5}
xform = function(input, w1, w2) {
    sig(input %*% w1) %*% w2
}

### The final data is 8 dimensional, but only looks complicated
train.data = xform(raw.train, w1, w2)
test.data = xform(raw.test, w1, w2)

### The encoder simply finds the nearest centroids
closest = function(v, k) {
    distances = rowSums(k$centers^2) + sum(v^2) - 2 * k$centers %*% v
    close = min(distances)
    which(distances == close)
}

encode = function(data, k) {
    apply(data, 1, function(v){closest(v, k)})
}

decode = function(kx, k) {
    as.matrix(k$centers[kx,])
}

reconstruction.error = function(data, k) {
    mean(sqrt(rowSums((data - decode(encode(data, k), k))^2)))
}

k.10 = kmeans(train.data, centers=10, nstart=10, iter.max=50)
k.20 = kmeans(train.data, centers=20, nstart=10, iter.max=50)
k.50 = kmeans(train.data, centers=50, nstart=10, iter.max=50)
k.100 = kmeans(train.data, centers=100, nstart=10, iter.max=50)
k.200 = kmeans(train.data, centers=200, nstart=10, iter.max=50)
k.500 = kmeans(train.data, centers=500, nstart=10, iter.max=50)
k.1000 = kmeans(train.data, centers=1000, nstart=5, iter.max=50)
k.2000 = kmeans(train.data, centers=2000, nstart=3, iter.max=50)

x.10 = reconstruction.error(train.data, k.10)
x.20 = reconstruction.error(train.data, k.20)
x.50 = reconstruction.error(train.data, k.50)
x.100 = reconstruction.error(train.data, k.100)
x.200 = reconstruction.error(train.data, k.200)
x.500 = reconstruction.error(train.data, k.500)
x.1000 = reconstruction.error(train.data, k.1000)
x.2000 = reconstruction.error(train.data, k.2000)

y.10 = reconstruction.error(test.data, k.10)
y.20 = reconstruction.error(test.data, k.20)
y.50 = reconstruction.error(test.data, k.50)
y.100 = reconstruction.error(test.data, k.100)
y.200 = reconstruction.error(test.data, k.200)
y.500 = reconstruction.error(test.data, k.500)
y.1000 = reconstruction.error(test.data, k.1000)
y.2000 = reconstruction.error(test.data, k.2000)

pdf(file="points.pdf", width=5, height=5)
plot(c(10,20,50,100,200,500,1000, 2000), 
     c(x.10, x.20, x.50, x.100, x.200, x.500, x.1000, x.2000),
     ylab="Error", xlab="Centroids", type='b',
     ylim=c(0,2), lwd=2,
     main="Reconstruction error for random points"
     )
    

lines(c(10,20,50,100,200,500,1000,2000), 
      c(y.10, y.20,  y.50,  y.100,  y.200,  y.500,  y.1000,  y.2000),
      col='red', lwd=2, type='b')

legend(1000, 1.9, legend=c("Training data", "Held-out data"), col=c("black", "red"), pch=21, lwd=2)
dev.off()

plot(inv~exp(log(n)*0.33333),x,type='b', xlab="k", ylab="1/error", main="Error is approximately cube root of k",
     lwd=2, ylim=c(0,3), xlim=c(0,13))
abline(lm(inv ~ rootK, x[1:4,]), lty=3)
