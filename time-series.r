# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set.seed(4)

## Pad with zeros on both ends
pad = function(x, pad.size=500) {
    c(rep(0,pad.size), x, rep(0,pad.size))
}

## Create a randomized impulse by creating white noise, down-filtering and windowing
impulse = function(n=32, f=n/2, pad.size=500) {
    t = 1:(2*n) - n - 1 + 1e-8
    v = sin(t/f * pi)/t * cos(t/2/n*pi)^2
    n = pad(rnorm(length(t)), n * 3)
    convolve(n,v, type="filter")
}

## Encode a windowed time-series using a specified dictionary
encode = function(win, dict) {
    ## largest dot product is most aligned centroid, size gives window scale
    dot = win %*% dict
    ## return nearest centroid in column 1, scale in column 2
    t(apply(dot, 1, function(v) {
        mx = max(abs(v))
        k = which(abs(v)==mx)[1]
        c(k, v[k])
    }))
}

## Reconstructs a signal from an encoded form
decode = function(data, dict) {
    n = dim(data)[1]
    r = rep(0, (n+1)*16)
    ## step a window at a time
    for (i in 1:n) {
        k = data[i,1]
        v = data[i,2]
        off = (i-1)*16
        ## add in current centroid after scaling and shifting it
        r[off + 1:32] = r[off + 1:32] + dict[,k] * v
    }
    r
}

## Computes the reconstruction error by reconstructing and measuring
reconstruction.error = function(data, dict) {
    rx = window(data)
    kx = encode(rx, dict)
    rz = decode(kx, dict)
    n = length(rz)
    mean(abs(data[15 + 1:n] - rz))    
}

## Generates a randomized time series
generate = function(windows, v.1, v.2) {
    ## signal is original all zeros
    r = rep(0, 16 * windows)
    t = 0
    while (t < length(r) - length(v.1)) {
        ## next pulse start point is exponentially distributed
        t = t + -0.5 * length(v.1) * log(runif(1))
        ## flip a coin to pick which pulse we want
        flip = rnorm(1) > 0
        n1 = floor(t)
        n2 = n1 + length(v.1) - 1
        if (n2 > length(r)) {
            return(r)
        }
        r[n1:n2] = r[n1:n2] + v.1*flip + v.2*(1-flip)
    }
    r
}

## Vector magnitude
mag = function(v) {
    sqrt(sum(v^2))
}

## Stack up a time series as overlapping window segments
window = function(data, window.size=32) {
    ## how many windows?
    n = floor(2 * length(data)/window.size)
    ## the window shape
    w = sin((0:(window.size-1))/window.size * pi)^2
    ## for each row index, apply the window function to a sub-sequence
    t(apply(matrix(1:(n-window.size/2), ncol=1), 1, function(i) {
        n = i*16
        w * data[n:(n+window.size-1)]
    }))
}

## Generate the data
v.1 = impulse(64, 8)
v.2 = impulse(64, 8)

training.data = generate(10000, v.1, v.2)
test.data = generate(10000, v.1, v.2)

## Go ahead and window once to help with all the clustering
rx = t(apply(window(training.data), 1, function(v){v/(1e-100+mag(v))}))

k.10 = kmeans(rx, centers=10, nstart=10)
c.10 = apply(k.10$centers,1,function(v){v/mag(v)})
x.10 = reconstruction.error(training.data, c.10)
y.10 = reconstruction.error(test.data, c.10)
print(10)

k.100 = kmeans(rx, centers=100, nstart=10, iter.max=20)
c.100 = apply(k.100$centers,1,function(v){v/(1e-100 + mag(v))})
x.100 = reconstruction.error(training.data, c.100)
y.100 = reconstruction.error(test.data, c.100)
print(100)

k.200 = kmeans(rx, centers=200, nstart=10, iter.max=30)
c.200 = apply(k.200$centers,1,function(v){v/(1e-100 + mag(v))})
x.200 = reconstruction.error(training.data, c.200)
y.200 = reconstruction.error(test.data, c.200)
print(200)

k.500 = kmeans(rx, centers=500, nstart=10, iter.max=30)
c.500 = apply(k.500$centers,1,function(v){v/(1e-100 + mag(v))})
x.500 = reconstruction.error(training.data, c.500)
y.500 = reconstruction.error(test.data, c.500)
print(500)

k.1000 = kmeans(rx, centers=1000, nstart=5, iter.max=10)
c.1000 = apply(k.1000$centers,1,function(v){v/(1e-100 + mag(v))})
x.1000 = reconstruction.error(training.data, c.1000)
y.1000 = reconstruction.error(test.data, c.1000)
print(1000)

k.2000 = kmeans(rx, centers=2000, nstart=5, iter.max=10)
c.2000 = apply(k.2000$centers,1,function(v){v/(1e-100 + mag(v))})
x.2000 = reconstruction.error(training.data, c.2000)
y.2000 = reconstruction.error(test.data, c.2000)
print(2000)


reconstruct.plot = function () {
    plot(test.data[1:2000], type='l', xlab="Time", ylab=NA, main="Time series training data (first 2000 samples)",
         lwd=3)
    rex = window(test.data)
    re.data = decode(encode(rex, c.1000), c.1000)
    lines(15 + 1:2000, re.data[1:2000], col='red')
    lines(15 + 1:2000, test.data[15+1:2000]-re.data[1:2000]-1, col='blue')
    legend(800, 2.2, legend=c("Test data", "Reconstruction", "Error"), col=c("black", "red", "blue"), pch=21, lty=1, cex=0.8)
}

time.series.error.plot = function() {
    plot(c(10,100,200,500,1000,2000), c(x.10, x.100, x.200, x.500, x.1000, x.2000), 
         type='b', lwd=2, xlab="Centroids", ylab="MAV Error",
         ylim=c(0,0.15),
         main="Reconstruction error for time-series data")

    lines(c(10,100,200,500,1000,2000), c(y.10, y.100, y.200, y.500, y.1000, y.2000), 
          type='b', lwd=2, col='red')

    legend(1000, 0.15, legend=c("Training data", "Held-out data"), col=c("black", "red"), pch=21, lwd=2)
}

time.series.plot = function() {
    plot(training.data[1:2000], type='l', xlab="Time", ylab=NA, main="Time series training data (first 2000 samples)")
}

pdf(file="time-series-errors.pdf", width=5, height=5)
time.series.error.plot()
dev.off()
                                             
png(file="images/time-series-errors.png", width=400, height=400)
time.series.error.plot()
dev.off()
                                             
pdf(file="time-series.pdf", width=5, height=5)
time.series.plot()
dev.off()

png(file="images/time-series.png", width=400, height=400)
time.series.plot()
dev.off()

pdf(file="time-series-reconstruction.pdf", width=5, height=5)
reconstruct.plot()
dev.off()

png(file="images/time-series-reconstruction.png", width=400, height=400)
reconstruct.plot()
dev.off()
