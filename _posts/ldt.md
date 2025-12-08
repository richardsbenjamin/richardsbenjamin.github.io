## An Introduction to Large Deviations Theory


## Example

Consider a sequence of $n$ independent and identically distributed (i.i.d.) random bits, $\mathbf{b} = (b_1, b_2, \ldots, b_n)$. Each bit $b_i$ is a Bernoulli trial, taking the value $1$ (Heads) or $0$ (Tails) with equal probability, $P(b_i=1) = P(b_i=0) = 1/2$.

The empirical mean or sample fraction of $1$'s in the sequence $\mathbf{b}$ is defined as:

$$R_n = \frac{1}{n}\sum_{i=1}^n b_i$$

$R_n$ represents the proportion of Heads in $n$ flips. Since the flips are unbiased, we expect $R_n$ to be close to $1/2$ for large $n$ (by the Law of Large Numbers). We want to find the probability $P(R_n = r)$ that this proportion equals a specific rational value $r \in \{0, 1/n, 2/n, \ldots, 1\}$.

Since each of the $2^n$ possible sequences $\mathbf{b} \in \{0, 1\}^n$ has the same probability $P(\mathbf{b}) = 2^{-n}$, the total probability $P(R_n = r)$ is simply the number of sequences that yield this ratio, multiplied by the probability of a single sequence.

The condition $R_n = r$ means the sequence $\mathbf{b}$ must contain exactly $k = nr$ ones (Heads). The number of such distinct sequences is a standard problem in combinatorics, given by the binomial coefficient:

$$\mathcal{N} = \binom{n}{k} = \binom{n}{nr} = \frac{n!}{(nr)!(n - nr)!}$$

The total probability is then:

$$P(R_n = r) = \sum_{\mathbf{b}: R_n(\mathbf{b})=r} P(\mathbf{b}) = \mathcal{N} \cdot 2^{-n}$$
$$P(R_n = r) = \frac{1}{2^n}\frac{n!}{(rn)!(n(1-r))!} $$

For Large Deviation Theory, we are primarily interested in the behavior of this probability as the sequence length $n$ becomes very large ($n \to \infty$). To simplify the expression for large $n$, we use Stirling's approximation for the factorials. Specifically, we use the simplified version that focuses on the dominant exponential component: $k! \approx k^k e^{-k}$.

Applying this to the three factorials:
- $n! \approx n^n e^{-n}$
- $(rn)! \approx (rn)^{rn} e^{-rn}$
- $(n(1-r))! \approx (n(1-r))^{n(1-r)} e^{-n(1-r)}$
  
Hence,

$$P(R_n = r) \approx \frac{1}{2^n} \cdot \frac{n^n e^{-n}}{(rn)^{rn} e^{-rn} \cdot (n(1-r))^{n(1-r)} e^{-n(1-r)}}$$

Grouping the terms, the exponential factors neatly cancel:

$$\frac{e^{-n}}{e^{-rn} e^{-n(1-r)}} = e^{-n - (-rn - n(1-r))} = e^0 = 1$$

Which simplifies the approximation to:

$$P(R_n = r) \approx \frac{1}{2^n} \cdot \frac{n^n}{(rn)^{rn} (n(1-r))^{n(1-r)}}$$

Next, we factor out $n^n$ from the denominator term:

$$\begin{aligned} (rn)^{rn} (n(1-r))^{n(1-r)} &= n^{rn} r^{rn} \cdot n^{n(1-r)} (1-r)^{n(1-r)} &= n^{rn + n(1-r)} \cdot r^{rn} (1-r)^{n(1-r)} &= n^n \cdot r^{rn} (1-r)^{n(1-r)} \end{aligned}$$.

The $n^n$ terms then cancel:

$$P(R_n = r) \approx \frac{1}{2^n} \cdot \frac{n^n}{n^n \cdot r^{rn} (1-r)^{n(1-r)}}$$
$$P(R_n = r) \approx \frac{1}{2^n r^{rn} (1-r)^{n(1-r)}}$$

To reveal the characteristic LDT form, we take the natural logarithm and isolate $n$ as a factor:

$$\ln \left[ P(R_n = r) \right] \approx - \ln \left[ 2^n r^{rn} (1-r)^{n(1-r)} \right]$$
$$\ln \left[ P(R_n = r) \right] \approx - \left[ n \ln 2 + rn \ln r + n(1-r) \ln (1-r) \right]$$
$$\ln \left[ P(R_n = r) \right] \approx -n \left[ \ln 2 + r \ln r + (1-r) \ln (1-r) \right]$$

Finally, exponentiating both sides yields the canonical LDT form:

$$P(R_n = r) \approx e^{-nI(r)}$$

where the function $I(r)$ is defined as:

$$I(r) = \ln 2 + r \ln r + (1-r) \ln (1-r)$$

The equation $P(R_n = r) \approx e^{-nI(r)}$ embodies the Large Deviation Principle. This result is the essence of LDT.

There are a couple of things to note about this. First, the exponent’s components are the key elements governing this probability. We have $n$, the system size (the number of bits/trials/flips), and what’s called the rate function $I(r)$. 

The fact that the probability is dominated by a term of the form $e^{-n \times (\text{something positive})}$ immediately signals a rapid, exponential decay in probability as the system size $n$ increases, unless (and this is important) $I(r)$ is zero.

The Rate Function, $I(r)$, is the engine that dictates this probability behavior, and its shape is paramount. It is positive and convex as seen in Fig. 1, and possesses a unique global minimum. For our specific example (random bits generated with equal probability, $p=1/2$), the minimum of $I(r)$ occurs precisely at $r = 1/2$. At this minimum, the value is $I(r=1/2) = 0$.

So, when $r=1/2$, the exponential approximation becomes $P(R_n = 1/2) \approx e^{-n \cdot 0} = e^0 = 1$.

This should be very intuitive. The proportion $r=1/2$ represents the most likely outcome, the expected average. Therefore, the probability distribution is dominated by the region around this value. While the true probability is not exactly 1, the formula effectively says that as $n$ grows, the vast majority of the probability mass is concentrated right at the average.

Another key takeaway connects the system size ($n$) to the probability of rare events ($r \neq 1/2$).

Large deviations from the expected average are exponentially suppressed as the system size increases. This is the core insight of Large Deviation Theory.

Any proportion $r$ that deviates from the average $1/2$ is considered a rare event (e.g., $r=0.9$ or $r=0.1$). For these values, we have $I(r) > 0$. When $r \neq 1/2$, the exponent contains a positive factor, $-nI(r)$. As $n$ increases (we flip the coin more times), the entire exponent $-nI(r)$ becomes increasingly negative. Consequently, the term $e^{-nI(r)}$ decreases exponentially towards zero. In contrast, for the expected event $r=1/2$, $I(r)$ remains zero, and $e^{-nI(r)}$ stays close to $1$.

When we randomly generate bits with equal probability, it is incredibly unlikely to observe a sequence where the majority of the outcomes all favor one side (e.g., 900 heads out of 1000 flips). The mathematical approximation confirms this intuition precisely: the probability of such an extreme outcome drops to zero exponentially fast as the number of trials ($n$) increases.

The fundamental insight of LDT is the formalisation of this intuitive notion: The likelihood of a random variable deviating significantly from its expected mean is governed by an exponential law involving the system size $n$ and the rate function $I(r)$.
