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


$$(rn)^{rn} (n(1-r))^{n(1-r)} = n^{rn} r^{rn} \cdot n^{n(1-r)} (1-r)^{n(1-r)}$$

$$\quad = n^{rn + n(1-r)} \cdot r^{rn} (1-r)^{n(1-r)}$$

$$\quad = n^n \cdot r^{rn} (1-r)^{n(1-r)}$$

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

There are a couple of things to note about this. First, the exponent’s components are the key elements governing this probability. We have $n$, the system size (the number of bits/trials/flips), and what’s called the rate function $I(r)$. Also, the fact that the probability is dominated by a term of the form $e^{-n \times (\text{something positive})}$ immediately signals a rapid, exponential decay in probability as the system size $n$ increases, unless (and this is important) $I(r)$ is zero.

The Rate Function, $I(r)$, is the engine that dictates this probability behavior, and its shape is paramount. It is positive and convex as seen in Fig. 1, and possesses a unique global minimum. For our specific example (random bits generated with equal probability, $p=1/2$), the minimum of $I(r)$ occurs precisely at $r = 1/2$. At this minimum, the value is $I(r=1/2) = 0$.

<figure>
  <img src="/assets/ldt_rate_function.png" alt="Graph" width="500" height="450" class="center-image">
  <figcaption class="figcaption-2">Fig. 1: Theoretical versus simulated rate function for random bits example.</figcaption>
</figure>

Note, when $r=1/2$, the exponential approximation becomes $P(R_n = 1/2) \approx e^{-n \cdot 0} = e^0 = 1$. This should be very intuitive. The proportion $r=1/2$ represents the most likely outcome, the expected average. Therefore, the probability distribution is dominated by the region around this value. While the true probability is not exactly 1, the formula effectively says that as $n$ grows, the vast majority of the probability mass is concentrated right at the average.

Another key takeaway connects the system size ($n$) to the probability of rare events ($r \neq 1/2$). Large deviations from the expected average are exponentially suppressed as the system size increases. This is the core insight of Large Deviation Theory.

Any proportion $r$ that deviates from the average $1/2$ is considered a rare event (e.g., $r=0.9$ or $r=0.1$). For these values, we have $I(r) > 0$. When $r \neq 1/2$, the exponent contains a positive factor, $-nI(r)$. As $n$ increases (we flip the coin more times), the entire exponent $-nI(r)$ becomes increasingly negative. Consequently, the term $e^{-nI(r)}$ decreases exponentially towards zero. In contrast, for the expected event $r=1/2$, $I(r)$ remains zero, and $e^{-nI(r)}$ stays close to $1$.

This is exactly what we observe in Fig. 2, which plots the theoretical PDF according to LDT for increasing n. As n increases, the density becomes increasingly narrower, and extreme events becomes exponentially unlikely. 

<figure>
  <img src="/assets/ldt_pdfs.png" alt="Graph" width="500" height="450" class="center-image">
  <figcaption class="figcaption-2">Fig. 2: Theoretical PDF for increasing n.</figcaption>
</figure>

When we randomly generate bits with equal probability, it is incredibly unlikely to observe a sequence where the majority of the outcomes all favor one side (e.g., 900 heads out of 1000 flips). The mathematical approximation confirms this intuition precisely: the probability of such an extreme outcome drops to zero exponentially fast as the number of trials ($n$) increases.

## Theory

The cornerstone of large deviation theory is the pervasive exponential approximation encountered in systems involving many random variables. This approximation forms the basis for defining the Large Deviation Principle (LDP).

### Defining the Large Deviation Principle
A basic scaling law of the form $P_n \approx e^{-nI}$, where $P_n$ is a probability, $n$ is a large parameter (often the number of variables or size of the system), and $I$ is a positive constant, is formally known as a large deviation principle. 

To formalise this, let $A_n$ be a continuous random variable indexed by $n$, with a probability density function (PDF) denoted by $f_n(a)$. We say that the probability $P(A_n \in B) = \int_B f_n(a) da$ satisfies a Large Deviation Principle with rate $I_B$ if the following limit exists:

$$\lim_{n \to \infty} -\frac{1}{n} \ln P(A_n \in B) = I_B$$

This mathematical statement captures the meaning of the approximation $P(A_n \in B) \approx e^{-nI_B}$. It asserts that the dominant behavior of the probability is a decaying exponential in $n$.

A more detailed LDP statement often focuses on the behavior of the PDF itself. We say that $A_n$ satisfies an LDP with rate function $I(a)$ if its density function behaves asymptotically as:
$$f_n(a) \approx e^{-nI(a)}$$

- **Non-trivial Rate**: The cases of interest in large deviation theory are those for which the limit $I_B$ is non-trivial, meaning $0 < I_B < \infty$.
- **Super-Exponential Decay**: If the limit does not exist, or if the probability $P(A_n \in B)$ decays faster than any $e^{-na}$ (for $a>0$), we say the decay is super-exponential and set $I_B = \infty$.
- **Slow Decay**: If $P(A_n \in B)$ decays slower than $e^{-na}$ (for $a>0$), the limit $I_B$ will be $0$.

The practical goal of large deviation theory involves two core problems:
- **Existence**: Establishing that an LDP exists for a given random variable $A_n$.
- **Derivation**: Deriving the explicit mathematical expression for the associated rate function, $I(a)$.
  
While direct calculation of the probability distribution and use of asymptotic formulas (like Stirling's approximation) can solve these problems in some cases, a more powerful and general method is required for complex systems. This method is provided by the Gärtner-Ellis Theorem.

### The Gärtner-Ellis Theorem
The Gärtner-Ellis Theorem provides a general calculation path by connecting the LDP to a function derived from the moment-generating function.

Consider a real, continuous random variable $A_n$ with PDF $f_n(a)$. We first define its Scaled Cumulant Generating Function (SCGF), $\lambda(k)$, by the following limit:

$$\lambda(k) = \lim_{n \to \infty} \frac{1}{n} \ln \langle e^{nk A_n} \rangle, \quad k \in \mathbb{R}$$

where $\langle e^{nk A_n} \rangle$ is the expectation value, which is defined using the PDF as:

$$\langle e^{nk A_n} \rangle = \int_{\mathbb{R}} e^{nk a} f_n(a) da$$

The Gärtner-Ellis Theorem states that if the SCGF, $\lambda(k)$, exists and is differentiable for all $k \in \mathbb{R}$, then $A_n$ satisfies a large deviation principle, meaning its PDF has the asymptotic behavior:

$$f_n(a) \approx e^{-nI(a)},$$

with the rate function $I(a)$ given by the Legendre-Fenchel Transform of $\lambda(k)$:

$$I(a) = \sup_{k \in \mathbb{R}}\{ka - \lambda(k)\}$$

Where, the $\sup$ (supremum) is the [supremum](https://en.wikipedia.org/wiki/Infimum_and_supremum), an extension of the maximum function.

In summary, the theorem says that when the SCGF, $\lambda(k)$, is differentiable, the random variable $A_n$ obeys an LDP with a rate function $I(a)$ given by the Legendre-Fenchel transform of $\lambda(k)$.

We won’t be presenting a full derivation here. Instead, we are going to gain a more intuitive understanding of why the Legendre-Fenchel transform appears by assuming the LDP holds and examining the consequences.

1. **Assume LDP for the PDF**: Start by assuming the LDP approximation for the density function:
$$f_n(a) \approx e^{-nI(a)}$$
2. **Substitute into Expectation**: Substitute this into the expectation value integral:
   
$$\langle e^{nk A_n} \rangle \approx \int_{\mathbb{R}} e^{n k a} e^{-n I(a)} da = \int_{\mathbb{R}} e^{-n [I(a) - k a]} da$$

3. **Apply Laplace's Approximation**: For large $n$, integrals of this form are dominated by the maximum value of the integrand's exponent. This maximum is found by locating the supremum of $(ka - I(a))$. This process is known as Laplace's Approximation (or the saddle-point approximation).
Applying this approximation yields:

$$\langle e^{nk A_n} \rangle \approx \exp \{ n \sup_{a \in \mathbb{R}} \left\\{ k a - I(a) \right\\} \}$$

4. **Derive $\lambda(k)$**: Now, substitute this result back into the definition of the SCGF:
   
$$\lambda(k) = \lim_{n \to \infty} \frac{1}{n} \ln \langle e^{nk A_n} \rangle = \sup_{a \in \mathbb{R}} \left\\{ k a - I(a) \right\\}$$

5. **Inversion via Self-Duality**: We now have $\lambda(k)$ as the Legendre-Fenchel transform of $I(a)$. The final step is to solve for $I(a)$ in terms of $\lambda(k)$. A key property of the Legendre-Fenchel transform is that it is self-inverse (or involutive) when the functions involved are appropriately differentiable and convex.
Assuming the differentiability of $\lambda(k)$, the self-inversion gives:

$$I(a) = \sup_{k \in \mathbb{R}} \left\\{ k a - \lambda(k) \right\\}$$

This heuristic derivation illustrates two crucial points. First, the appearance of the Legendre-Fenchel transform is a natural consequence of applying Laplace's approximation to the moment-generating function integral. Second, the Gärtner-Ellis Theorem is fundamentally a consequence of the large deviation principle combined with Laplace's approximation, provided the convexity conditions are met.






