---
title: Multivariable Calculus
tags:
  - calculus
  - foundations
date: 2026-11-12
draft: false
prev: "[[Index (Probability and Linear Algebra)]]"
next: "[[Column Space and Rank]]"
---

> **Prerequisites:** single-variable calculus (derivatives and integrals of familiar functions), vectors, dot products, and multiplying a matrix by a vector. Linear algebra beyond that is introduced as needed, with links forward.
>
> **Series:** this is part 1 of 8. **Multivariable Calculus** → [[Column Space and Rank]] → [[Elimination and the Nullspace]] → [[The Four Fundamental Subspaces]] → [[Orthogonality and Least Squares]] → [[Eigenvalues and Eigenvectors]] → [[First-Order Optimization]] → [[Five Factorizations of a Matrix]]. The probability lectures, starting with [[Probability Basics]], come next.
>
> **TL;DR:**
> - A derivative is the **best linear approximation** of a function near a point. In many variables that linear approximation is the **gradient** (for a scalar function) or the **Jacobian** (for a vector-valued map), and the chain rule says that the linear approximations of composed functions multiply.
> - The gradient points in the direction of steepest ascent and is perpendicular to the level sets. The **Hessian** holds the second derivatives; at a critical point its eigenvalue signs decide between minimum, maximum and saddle.
> - Matrix calculus is the same idea applied to vectors and matrices. Expanding $f(\mathbf{x}+\mathbf{h})$ and reading off the term linear in $\mathbf{h}$ gives the gradient of every function used in the series, including least squares (the normal equations) and log-sum-exp (the softmax).
> - For **convex** functions, every local minimum is global and a zero gradient certifies optimality. With constraints, the test becomes the **Lagrange** and **KKT** conditions, and $\nabla f$ must line up with the constraint gradients.
> - Changing variables in an integral multiplies by $\lvert\det\mathbf{J}\rvert$, the factor by which the map scales volume. Polar coordinates turn the Gaussian integral into $\sqrt\pi$.
>
> **Notation in this lecture:** vectors are bold lowercase ($\mathbf{x}\in\mathbb{R}^n$) and matrices bold uppercase. A function of several variables is $f:\mathbb{R}^n\to\mathbb{R}$, and in two dimensions we write $f(x,y)$. Gradients are **column vectors** with the same shape as the variable (layout convention in [[#Matrix calculus]]).

## Motivation

Every model in the series is trained by changing many numbers at once so that a single number, the loss, goes down. A linear regression has $D$ weights, a neural network has millions, and the loss $\mathcal{L}(\mathbf{w})$ is a function of all of them together. To decrease it we need to know how it responds to a small change in **every direction at once**: that is the gradient. To know whether we have reached the bottom of a valley, a ridge or a mountain pass, we need second derivatives: the Hessian. Constraints (probabilities must sum to one, a margin must be at least one) bring in Lagrange multipliers, and densities of transformed random variables bring in Jacobian determinants.

None of this needs new ideas beyond single-variable calculus. It needs one idea, used carefully: **near a point, a smooth function looks linear**, and a little further away it looks quadratic. The plan of this lecture is to make that idea precise in one variable, then carry it over to many variables piece by piece, with small examples worked by hand at every step. All the numbers in the examples were checked by a script. For a longer treatment in the same spirit, see Chapters 5 and 7 of *Mathematics for Machine Learning*.[^mml2020]

## Single-variable recap

### The derivative is the best linear approximation

The derivative of $f$ at $a$ is the limit of slopes of secant lines,

$$
f'(a)=\lim_{h\to0}\frac{f(a+h)-f(a)}{h}.
$$

There is a second way to say the same thing, and it is the one that generalizes. Rearranging the definition,

$$
f(a+h)=f(a)+f'(a)\,h+o(h),
\tag{1}
$$

where $o(h)$ stands for an error that shrinks **faster than** $h$: $o(h)/h\to0$. So near $a$ the function is a straight line, the tangent line, plus an error that is negligible compared with the step. No other slope has this property: with any slope $m\ne f'(a)$ the error would be about $(f'(a)-m)h$, which is not small compared with $h$. The derivative is the slope of **the best linear approximation**.

> [!example] Example: $x^2$ near $3$
> For $f(x)=x^2$, expand exactly: $(3+h)^2=9+6h+h^2$. The constant is $f(3)=9$, the coefficient of $h$ is $f'(3)=6$, and the error is $h^2$, which is indeed $o(h)$. With $h=0.1$, the tangent line predicts $9.6$ and the truth is $9.61$. Halving $h$ halves the linear term but **quarters** the error. That is what "best linear approximation" means in practice.

### Rules: product and chain

Two rules do most of the work. The **product rule** is $(uv)'=u'v+uv'$: if $u$ and $v$ both change a little, the area $uv$ of a rectangle with sides $u$ and $v$ grows by two thin strips (the tiny corner piece is second order). For example, $\frac{d}{dx}\,xe^x=e^x+xe^x=(1+x)e^x$.

The **chain rule** handles compositions. If $y=g(u)$ and $u=h(x)$, then

$$
\frac{dy}{dx}=\frac{dy}{du}\,\frac{du}{dx}=g'\big(h(x)\big)\,h'(x).
$$

Rates of change multiply. If $u$ moves 3 times as fast as $x$ and $y$ moves 2 times as fast as $u$, then $y$ moves 6 times as fast as $x$. In terms of (1): a small step $h$ in $x$ becomes a step of about $h'(x)\,h$ in $u$, which becomes a step of about $g'(u)\,h'(x)\,h$ in $y$. For example, $\frac{d}{dx}\sin(x^2)=\cos(x^2)\cdot2x$, and $\frac{d}{dx}e^{-x^2/2}=-x\,e^{-x^2/2}$. The multivariable chain rule below is exactly this sentence with matrices in place of numbers.

> [!tip] Intuition
> 3Blue1Brown's *Visualizing the chain rule and product rule* draws the "thin strips" of the product rule and the "nudges passing through" of the chain rule.[^3b1b-chain]

### Taylor's theorem to second order

The tangent line ignores how the slope itself changes. Adding one more term,

$$
f(a+h)=f(a)+f'(a)\,h+\tfrac12f''(a)\,h^2+o(h^2).
\tag{2}
$$

The $\frac12$ is there so that the quadratic on the right has the same second derivative as $f$: the second derivative of $\frac12ch^2$ is $c$. In general the Taylor polynomial matches the value, slope and curvature of $f$ at $a$.

> [!example] Example: computing $\sqrt{4.1}$ by hand
> Take $f(x)=\sqrt x$ at $a=4$. Then $f(4)=2$, $f'(x)=\frac{1}{2\sqrt x}$ gives $f'(4)=\frac14$, and $f''(x)=-\frac{1}{4x^{3/2}}$ gives $f''(4)=-\frac{1}{32}$. With $h=0.1$:
> - linear: $2+\frac14(0.1)=2.025$;
> - quadratic: $2.025-\frac12\cdot\frac{1}{32}(0.1)^2=2.025-0.00015625=2.0248438$;
> - true value: $\sqrt{4.1}=2.0248457$.
>
> The linear model is right to 3 decimal places and the quadratic to 5. The correction is negative because $\sqrt x$ bends **down**: $f''<0$ (Figure 1a).

### Critical points and the second-derivative test

At a minimum or maximum in the interior of the domain, the tangent line is flat: $f'(a)=0$. Such a point is a **critical point**. Near it, (2) loses its linear term,

$$
f(a+h)-f(a)\approx\tfrac12f''(a)\,h^2,
$$

so the sign of $f''(a)$ decides what happens. If $f''(a)>0$, then $f$ rises on both sides: a **local minimum**. If $f''(a)<0$, it falls on both sides: a **local maximum**. If $f''(a)=0$, the quadratic term vanishes too and the test says nothing.

> [!example] Example: $f(x)=x^3-3x$
> Here $f'(x)=3x^2-3=3(x-1)(x+1)$, so the critical points are $x=\pm1$. And $f''(x)=6x$:
> - at $x=1$, $f''=6>0$: a local minimum, with $f(1)=-2$;
> - at $x=-1$, $f''=-6<0$: a local maximum, with $f(-1)=2$.
>
> Neither is global: $f\to\pm\infty$ as $x\to\pm\infty$ (Figure 1b). The test is **local**.

When $f''(a)=0$, anything can happen. At $0$, $x^3$ has neither a maximum nor a minimum (an inflection point), $x^4$ has a minimum and $-x^4$ a maximum, and all three have $f'(0)=f''(0)=0$. Higher derivatives are needed to decide.

### Why the second derivative is curvature

$f''$ is the rate of change of the slope. Where $f''>0$, the slope increases as we move right, so the graph bends upwards like a cup. Where $f''<0$ the slope decreases and the graph bends down like a cap. At a critical point the graph looks like the parabola $\frac12f''(a)h^2$, so $\lvert f''(a)\rvert$ also says **how sharply** it bends: large $\lvert f''\rvert$ is a narrow valley, small $\lvert f''\rvert$ a wide, flat one. (The geometric curvature of a graph is $\lvert f''\rvert/(1+f'^2)^{3/2}$, which equals $\lvert f''\rvert$ exactly where the tangent is flat.) This is the quantity that will control how fast gradient descent converges in [[First-Order Optimization]]: flat directions are slow, sharply curved ones force small steps.

![[Probability and Linear Algebra/images/multivariable-calculus-single-variable.svg]]
*Figure 1. (a) The tangent line and the quadratic Taylor model of $\sqrt x$ at $x=4$. Near the point both are excellent; further away the quadratic, which knows that the graph bends down, stays closer. (b) $f(x)=x^3-3x$ has a local maximum where it bends down ($f''<0$, left half) and a local minimum where it bends up ($f''>0$, right half).*

> [!tip] Intuition
> 3Blue1Brown's *Taylor series* builds the quadratic and higher Taylor approximations one term at a time.[^3b1b-taylor]

## Functions of several variables

A function $f:\mathbb{R}^n\to\mathbb{R}$ assigns a number to each point $\mathbf{x}=(x_1,\dots,x_n)$. In machine learning the input is usually a parameter vector and the output a loss. With two inputs we can draw the **graph** $z=f(x,y)$, a surface over the plane (Figure 2a). With more inputs we cannot draw the graph, so we need a picture that survives in higher dimensions: level sets.

The **level set** of $f$ at height $c$ is the set of points where $f$ takes the value $c$:

$$
\{\mathbf{x}:\ f(\mathbf{x})=c\}.
$$

For $n=2$ the level sets are curves, and drawing several of them gives a **contour plot**, exactly like the contour lines on a hiking map. Where contours are close together the ground is steep; where they are far apart it is flat. For $n=3$ the level sets are surfaces, and in general they are $(n-1)$-dimensional.

> [!example] Example: the running example $f(x,y)=x^2+2y^2$
> The level set $x^2+2y^2=c$ (for $c>0$) is an ellipse, $\frac{x^2}{c}+\frac{y^2}{c/2}=1$, with semi-axes $\sqrt c$ along $x$ and $\sqrt{c/2}$ along $y$. The ellipses are squashed in the $y$ direction: $f$ grows twice as fast in $y$ as in $x$, so the contours crowd together vertically (Figure 2b). The level set at $c=0$ is the single point at the origin, the minimum, and for $c<0$ it is empty.

We will reuse this example for the gradient, the tangent plane and convexity. The quadratic shape is no accident: the second-order Taylor expansion below says that **every** smooth function looks like a quadratic near a minimum, so these ellipses are what every loss landscape looks like at the bottom of a valley.

## Partial derivatives and the gradient

### Partial derivatives

The simplest way to differentiate a function of several variables is to let only **one** of them move. The **partial derivative** with respect to $x$ treats $y$ as a constant:

$$
\frac{\partial f}{\partial x}(x,y)=\lim_{h\to0}\frac{f(x+h,y)-f(x,y)}{h}.
$$

Geometrically, it is the slope of the curve obtained by slicing the graph with the plane $y=\text{const}$. Computing it needs nothing new: freeze the other variables and use the single-variable rules. For $f(x,y)=x^2y^3$,

$$
\frac{\partial f}{\partial x}=2xy^3,\qquad\frac{\partial f}{\partial y}=3x^2y^2.
$$

For the running example $f=x^2+2y^2$: $\partial f/\partial x=2x$ and $\partial f/\partial y=4y$.

### Differentiability and the gradient

The partial derivatives only look along the coordinate axes. What about other directions? Copy the idea of (1): $f$ is **differentiable** at $\mathbf{a}$ if, near $\mathbf{a}$, it is a linear function plus a negligible error,

$$
f(\mathbf{a}+\mathbf{h})=f(\mathbf{a})+\mathbf{g}^\top\mathbf{h}+o(\lVert\mathbf{h}\rVert)
\tag{3}
$$

for some vector $\mathbf{g}$. Taking $\mathbf{h}=h\,\mathbf{e}_i$ along the $i$-th axis shows that $g_i$ must be the $i$-th partial derivative, so $\mathbf{g}$ is the vector of partial derivatives, the **gradient**:

$$
\nabla f(\mathbf{a})=\begin{bmatrix}\partial f/\partial x_1\\\vdots\\\partial f/\partial x_n\end{bmatrix}.
$$

For the running example, $\nabla f(x,y)=(2x,4y)^\top$, and at the point $(1,1)$ the gradient is $(2,4)^\top$.

> [!warning] Partial derivatives can exist without the function being differentiable
> Take $f(x,y)=\dfrac{xy}{x^2+y^2}$ with $f(0,0)=0$. On both axes $f$ is identically 0, so both partial derivatives at the origin exist and equal 0. But along the diagonal $y=x$, $f=\frac12$ everywhere except at the origin. The function is not even continuous at $(0,0)$, let alone approximately linear. The partial derivatives see only two directions, and (3) asks about all of them. A convenient sufficient condition: if the partial derivatives exist and are **continuous** near $\mathbf{a}$, then $f$ is differentiable at $\mathbf{a}$. Every function in this series satisfies it wherever we differentiate.

### The directional derivative

The rate of change of $f$ when we move from $\mathbf{a}$ in the direction of a unit vector $\mathbf{u}$ is the **directional derivative**. Putting $\mathbf{h}=t\mathbf{u}$ into (3),

$$
D_\mathbf{u}f(\mathbf{a})=\lim_{t\to0}\frac{f(\mathbf{a}+t\mathbf{u})-f(\mathbf{a})}{t}=\nabla f(\mathbf{a})^\top\mathbf{u}.
\tag{4}
$$

So the gradient knows the slope in **every** direction: just take a dot product. The partial derivatives are the special cases $\mathbf{u}=\mathbf{e}_i$.

> [!example] Example: the slope towards the north-north-east
> For $f=x^2+2y^2$ at $(1,1)$, move in the direction $\mathbf{u}=(\frac35,\frac45)$, a unit vector. Then
>
> $$
> D_\mathbf{u}f(1,1)=(2,4)\cdot\big(\tfrac35,\tfrac45\big)=1.2+3.2=4.4.
> $$
>
> A step of length $0.01$ in that direction should raise $f$ by about $0.044$.

### The gradient is the direction of steepest ascent

Which direction gives the largest slope? By (4) and the cosine formula for the dot product,

$$
D_\mathbf{u}f=\nabla f^\top\mathbf{u}=\lVert\nabla f\rVert\,\lVert\mathbf{u}\rVert\cos\theta=\lVert\nabla f\rVert\cos\theta,
$$

where $\theta$ is the angle between $\mathbf{u}$ and $\nabla f$. This is largest when $\cos\theta=1$, that is, $\mathbf{u}=\nabla f/\lVert\nabla f\rVert$. Three facts follow at once:
- **the gradient points in the direction of steepest ascent**, and its length $\lVert\nabla f\rVert$ is the slope in that direction;
- the negative gradient is the direction of **steepest descent**, with slope $-\lVert\nabla f\rVert$; this is why gradient descent steps along $-\nabla f$;
- in directions with $\theta=90^\circ$ the slope is zero.

In the example, $\lVert(2,4)\rVert=\sqrt{20}\approx4.47$. The direction $(\frac35,\frac45)$ gave $4.4$, only slightly less, because it is only about $10^\circ$ away from the gradient.

### The gradient is perpendicular to the level sets

The directions of zero slope are the directions in which $f$ does not change, to first order. Those are the directions **along** the level set. So the gradient should be perpendicular to the level set through the point. Here is the argument.

> [!proof]- Proof that $\nabla f$ is perpendicular to level sets
> Let $\mathbf{c}(t)$ be any differentiable curve that stays inside the level set $\{f=c\}$ and passes through $\mathbf{a}$ at $t=0$. Then $f(\mathbf{c}(t))=c$ for all $t$, a constant, so its derivative is zero. By the chain rule (proved in the next section, and in this case a direct consequence of (3)),
>
> $$
> 0=\frac{d}{dt}f\big(\mathbf{c}(t)\big)\Big|_{t=0}=\nabla f(\mathbf{a})^\top\mathbf{c}'(0).
> $$
>
> So $\nabla f(\mathbf{a})$ is orthogonal to the velocity $\mathbf{c}'(0)$ of every curve in the level set, that is, to every tangent direction of the level set.

> [!example] Example: checking it by hand
> The level set of $f=x^2+2y^2$ through $(1,1)$ is the ellipse $x^2+2y^2=3$. Along it, $2x\,dx+4y\,dy=0$, so at $(1,1)$ a tangent direction satisfies $2\,dx+4\,dy=0$, for example $(dx,dy)=(2,-1)$. And indeed $(2,4)\cdot(2,-1)=4-4=0$. Figure 2b shows the gradient arrows crossing every contour at a right angle.

This is the picture to keep in mind for the rest of the series. Gradient descent walks downhill perpendicular to the contours. When the contours are long thin ellipses, "perpendicular to the contour" does not point at the minimum, and gradient descent zig-zags; [[First-Order Optimization]] makes that precise.

### Linear approximation and the tangent plane

Dropping the error term in (3) gives the **linear approximation**

$$
f(\mathbf{x})\approx f(\mathbf{a})+\nabla f(\mathbf{a})^\top(\mathbf{x}-\mathbf{a}).
$$

For $n=2$ its graph is the **tangent plane** to the surface $z=f(x,y)$ at $\mathbf{a}$: the plane that touches the surface there and has the same slope in every direction.

> [!example] Example: the tangent plane at $(1,1)$
> For $f=x^2+2y^2$ we have $f(1,1)=3$ and $\nabla f(1,1)=(2,4)$, so the tangent plane is
>
> $$
> z=3+2(x-1)+4(y-1).
> $$
>
> At $(1.1,\,0.9)$ it predicts $3+0.2-0.4=2.80$. The true value is $1.21+1.62=2.83$. The difference $0.03$ is second order: it comes from the curvature, which the Hessian below will capture.

![[Probability and Linear Algebra/images/multivariable-calculus-contours.svg]]
*Figure 2. (a) The graph of $f(x,y)=x^2+2y^2$, a bowl, with its level sets drawn on the floor. (b) The contour plot with normalized gradient directions (blue). At $(1,1)$ the gradient $(2,4)$ (red) is perpendicular to the tangent of the level set $f=3$ (green). The direction $\mathbf{u}=(\frac35,\frac45)$ (orange) is about $10^\circ$ from the gradient, so its slope, $4.4$, is close to the maximum, $\sqrt{20}\approx4.47$.*

## The Jacobian and the chain rule

### Vector-valued maps and the Jacobian

Many maps output several numbers: a change of coordinates, a layer of a neural network, a vector field. For $\mathbf{F}:\mathbb{R}^n\to\mathbb{R}^m$ with components $F_1,\dots,F_m$, each component has a gradient. Stacking the transposed gradients as rows gives the $m\times n$ **Jacobian matrix**

$$
\mathbf{J}_\mathbf{F}(\mathbf{a})=\begin{bmatrix}\nabla F_1(\mathbf{a})^\top\\\vdots\\\nabla F_m(\mathbf{a})^\top\end{bmatrix},\qquad\big(\mathbf{J}_\mathbf{F}\big)_{ij}=\frac{\partial F_i}{\partial x_j}.
$$

Row $i$ is "how output $i$ responds to each input", and column $j$ is "how all outputs respond to input $j$". Writing (3) for each component and stacking,

$$
\mathbf{F}(\mathbf{a}+\mathbf{h})=\mathbf{F}(\mathbf{a})+\mathbf{J}_\mathbf{F}(\mathbf{a})\,\mathbf{h}+o(\lVert\mathbf{h}\rVert).
$$

So the Jacobian is **the best linear map** approximating $\mathbf{F}$ near $\mathbf{a}$, exactly as $f'(a)$ is the best slope. For a scalar function ($m=1$) the Jacobian is the single row $\nabla f^\top$, and for a linear map $\mathbf{F}(\mathbf{x})=\mathbf{A}\mathbf{x}$ the Jacobian is $\mathbf{A}$ itself, everywhere.

> [!example] Example: a Jacobian by hand
> Let $\mathbf{F}(x,y)=\big(x^2y,\ 5x+\sin y\big)$. Differentiating each component with respect to each variable,
>
> $$
> \mathbf{J}_\mathbf{F}(x,y)=\begin{bmatrix}2xy&x^2\\5&\cos y\end{bmatrix},\qquad\mathbf{J}_\mathbf{F}(1,0)=\begin{bmatrix}0&1\\5&1\end{bmatrix}.
> $$
>
> Since $\mathbf{F}(1,0)=(0,5)$, a step $\mathbf{h}=(0.01,0.02)$ should give about $(0,5)+(0.02,\ 0.05+0.02)=(0.02,\ 5.07)$. The true value is $\mathbf{F}(1.01,0.02)=(0.020402,\ 5.069999)$.

### The chain rule is a product of Jacobians

Compose two maps, $\mathbf{x}\mapsto\mathbf{u}=\mathbf{F}(\mathbf{x})$ and $\mathbf{u}\mapsto\mathbf{G}(\mathbf{u})$. A small step $\mathbf{h}$ in $\mathbf{x}$ becomes a step of about $\mathbf{J}_\mathbf{F}\mathbf{h}$ in $\mathbf{u}$, and that becomes a step of about $\mathbf{J}_\mathbf{G}\,\mathbf{J}_\mathbf{F}\,\mathbf{h}$ in the output. The linear approximation of the composition is the composition of the linear approximations, and composing linear maps means multiplying matrices:

$$
\mathbf{J}_{\mathbf{G}\circ\mathbf{F}}(\mathbf{x})=\mathbf{J}_\mathbf{G}\big(\mathbf{F}(\mathbf{x})\big)\;\mathbf{J}_\mathbf{F}(\mathbf{x}).
\tag{5}
$$

This is the single-variable chain rule, "rates multiply", with matrices for rates. The shapes always fit: if $\mathbf{F}:\mathbb{R}^n\to\mathbb{R}^k$ and $\mathbf{G}:\mathbb{R}^k\to\mathbb{R}^m$, the product is $(m\times k)(k\times n)=m\times n$. Two special cases come up constantly:
- **A scalar function along a curve.** If $\mathbf{x}(t)$ is a curve and $f$ is scalar, then $\frac{d}{dt}f(\mathbf{x}(t))=\nabla f^\top\mathbf{x}'(t)=\sum_i\frac{\partial f}{\partial x_i}\frac{dx_i}{dt}$. Each input's rate of change, times how much $f$ cares about that input, summed.
- **A scalar function of new variables.** If $g(\mathbf{u})=f(\mathbf{F}(\mathbf{u}))$, then $\frac{\partial g}{\partial u_j}=\sum_i\frac{\partial f}{\partial x_i}\frac{\partial x_i}{\partial u_j}$; as a row vector, $\nabla g^\top=\nabla f^\top\mathbf{J}_\mathbf{F}$.

### Worked example: polar coordinates

Polar coordinates describe a point by its distance $r$ from the origin and its angle $\theta$:

$$
\mathbf{P}(r,\theta)=\begin{bmatrix}r\cos\theta\\r\sin\theta\end{bmatrix},\qquad\mathbf{J}_\mathbf{P}=\begin{bmatrix}\cos\theta&-r\sin\theta\\\sin\theta&r\cos\theta\end{bmatrix}.
$$

The first column, $\partial(x,y)/\partial r$, is the unit vector pointing away from the origin: increasing $r$ moves the point outwards at unit speed. The second column, $\partial(x,y)/\partial\theta$, is perpendicular to it and has length $r$: turning by a small angle moves a point at distance $r$ along an arc of length $r\,d\theta$. Its determinant, $r\cos^2\theta+r\sin^2\theta=r$, will reappear when we [[#Multiple integrals and change of variables|change variables in integrals]].

Now take $f(x,y)=xy$ and ask how it changes with $\theta$ at fixed $r$. The chain rule gives

$$
\begin{aligned}
\frac{\partial g}{\partial\theta}&=\frac{\partial f}{\partial x}\frac{\partial x}{\partial\theta}+\frac{\partial f}{\partial y}\frac{\partial y}{\partial\theta}\\
&=y\,(-r\sin\theta)+x\,(r\cos\theta)\\
&=r^2\big(\cos^2\theta-\sin^2\theta\big)=r^2\cos2\theta.
\end{aligned}
$$

Check by substituting first: $g(r,\theta)=r^2\cos\theta\sin\theta=\frac12r^2\sin2\theta$, whose $\theta$-derivative is $r^2\cos2\theta$. The same. On a circle, $xy$ is largest at $\theta=45^\circ$, where $\cos2\theta=0$, as it should be.

### Backpropagation is the chain rule

A neural network is a long composition, $\mathbf{x}\mapsto\mathbf{h}^{(1)}\mapsto\mathbf{h}^{(2)}\mapsto\cdots\mapsto\ell$, ending in a scalar loss. By (5) the gradient of the loss with respect to the input of any layer is a product of Jacobians,

$$
\nabla_{\mathbf{h}^{(1)}}\ell^\top=\underbrace{\frac{\partial\ell}{\partial\mathbf{h}^{(L)}}}_{1\times n_L}\;\frac{\partial\mathbf{h}^{(L)}}{\partial\mathbf{h}^{(L-1)}}\cdots\frac{\partial\mathbf{h}^{(2)}}{\partial\mathbf{h}^{(1)}}.
$$

The product can be evaluated in either order, but not at the same cost. Starting from the left, every intermediate result is a **row vector** (because the loss is a scalar), and multiplying a row vector by a matrix is cheap. Starting from the right would multiply large matrices together. Evaluating from the loss backwards is **reverse-mode differentiation**, better known as backpropagation; [[Neural Networks#Backpropagation|Neural Networks]] works it out layer by layer.

## The Hessian and second-order Taylor

### Second partial derivatives and the Hessian

Differentiating a partial derivative again gives second partial derivatives, such as $\frac{\partial^2f}{\partial x^2}$ and the **mixed** partial $\frac{\partial^2f}{\partial x\,\partial y}=\frac{\partial}{\partial x}\Big(\frac{\partial f}{\partial y}\Big)$. Collecting all of them in an $n\times n$ matrix gives the **Hessian**,

$$
\mathbf{H}_f(\mathbf{a})=\nabla^2f(\mathbf{a}),\qquad H_{ij}=\frac{\partial^2f}{\partial x_i\,\partial x_j}.
$$

It is the Jacobian of the gradient: row $i$ says how the $i$-th component of the gradient changes as we move.

> [!example] Example: the Hessian of $x^2y^3$
> From $f_x=2xy^3$ and $f_y=3x^2y^2$:
>
> $$
> \mathbf{H}_f=\begin{bmatrix}f_{xx}&f_{xy}\\f_{yx}&f_{yy}\end{bmatrix}=\begin{bmatrix}2y^3&6xy^2\\6xy^2&6x^2y\end{bmatrix},\qquad\mathbf{H}_f(1,2)=\begin{bmatrix}16&24\\24&12\end{bmatrix}.
> $$
>
> The two mixed partials came out equal: differentiating $2xy^3$ by $y$ and $3x^2y^2$ by $x$ both give $6xy^2$.

That is not luck. **Schwarz's theorem** (also called Clairaut's theorem) says that if the second partial derivatives are continuous near a point, the order of differentiation does not matter: $f_{xy}=f_{yx}$ there. So the Hessian of every smooth function is a **symmetric matrix**. Symmetry matters a great deal, because symmetric matrices have real eigenvalues and perpendicular eigenvectors ([[Eigenvalues and Eigenvectors#Symmetric matrices: S = QΛQᵀ|the spectral theorem]]), and that is what makes the classification of critical points below work.

### The quadratic model

To get the second-order Taylor expansion in many variables, reduce to one variable. Fix a direction $\mathbf{h}$ and look at $f$ along the line through $\mathbf{a}$: $g(t)=f(\mathbf{a}+t\mathbf{h})$. By the chain rule,

$$
g'(t)=\nabla f(\mathbf{a}+t\mathbf{h})^\top\mathbf{h},\qquad g''(t)=\mathbf{h}^\top\mathbf{H}_f(\mathbf{a}+t\mathbf{h})\,\mathbf{h}.
$$

(The second follows by applying the first to each component of the gradient.) The single-variable expansion (2) of $g$ at $t=0$, evaluated at $t=1$, then reads

$$
f(\mathbf{a}+\mathbf{h})=f(\mathbf{a})+\nabla f(\mathbf{a})^\top\mathbf{h}+\tfrac12\,\mathbf{h}^\top\mathbf{H}_f(\mathbf{a})\,\mathbf{h}+o(\lVert\mathbf{h}\rVert^2).
\tag{6}
$$

Compare with (2): the slope $f'$ became the gradient, and the curvature $f''$ became the Hessian, which now sits in a **quadratic form** $\mathbf{h}^\top\mathbf{H}\mathbf{h}$. The number $\mathbf{h}^\top\mathbf{H}\mathbf{h}/\lVert\mathbf{h}\rVert^2$ is the curvature of $f$ along the direction $\mathbf{h}$. Newton's method minimizes this quadratic model at each step; gradient descent uses only the linear part.

For the running example, $\mathbf{H}=\begin{bmatrix}2&0\\0&4\end{bmatrix}$ everywhere, so (6) is exact. The tangent plane at $(1,1)$ was off by $0.03$ at $\mathbf{h}=(0.1,-0.1)$, and the quadratic term is $\frac12(2\cdot0.01+4\cdot0.01)=0.03$, exactly the missing piece.

### Classifying critical points

A **critical point** (or stationary point) is a point where $\nabla f(\mathbf{a})=\mathbf{0}$: the tangent plane is flat. Then (6) loses its linear term and

$$
f(\mathbf{a}+\mathbf{h})-f(\mathbf{a})\approx\tfrac12\,\mathbf{h}^\top\mathbf{H}\,\mathbf{h}.
$$

The sign of the quadratic form, over all directions $\mathbf{h}$, decides what kind of point it is:
- $\mathbf{h}^\top\mathbf{H}\mathbf{h}>0$ for every $\mathbf{h}\ne\mathbf{0}$ (**positive definite** $\mathbf{H}$): $f$ rises in every direction, a **local minimum**;
- $\mathbf{h}^\top\mathbf{H}\mathbf{h}<0$ for every $\mathbf{h}\ne\mathbf{0}$ (**negative definite**): a **local maximum**;
- positive in some directions and negative in others (**indefinite**): a **saddle point**, like a mountain pass, uphill along the ridge and downhill across it;
- zero in some direction and never of the opposite sign (**semidefinite but singular**): the test is inconclusive, as when $f''=0$ in one variable.

In one variable, a critical point is a minimum or a maximum or flat. In two or more variables the saddle is new, and in high dimensions it is the typical case: a random symmetric matrix is much more likely to have eigenvalues of both signs than all of one sign.

### The 2×2 test: determinant and trace

For two variables the test can be done by hand. Write $\mathbf{H}=\begin{bmatrix}a&b\\b&c\end{bmatrix}$. If $a\ne0$, complete the square:

$$
\begin{aligned}
\mathbf{h}^\top\mathbf{H}\mathbf{h}&=ah_1^2+2bh_1h_2+ch_2^2\\
&=a\Big(h_1+\frac{b}{a}h_2\Big)^2+\frac{ac-b^2}{a}\,h_2^2.
\end{aligned}
$$

Both squares are non-negative, so everything depends on the signs of $a$ and of $\det\mathbf{H}=ac-b^2$:
- $\det\mathbf{H}>0$ and $a>0$: both terms positive, a **minimum**;
- $\det\mathbf{H}>0$ and $a<0$: both negative, a **maximum**;
- $\det\mathbf{H}<0$: the two terms have opposite signs, a **saddle**;
- $\det\mathbf{H}=0$: inconclusive.

When $\det\mathbf{H}>0$, $a$ and $c$ have the same sign, so the trace $a+c$ can be checked instead of $a$. In terms of eigenvalues this is no surprise: $\det\mathbf{H}=\lambda_1\lambda_2$ and $\operatorname{tr}\mathbf{H}=\lambda_1+\lambda_2$. A positive determinant means the eigenvalues have the same sign, and the trace tells which.

> [!example] Example: a minimum, a maximum and a saddle
> Each of these has a critical point at the origin (all are quadratics with no linear terms), and its Hessian is constant.
> - $f=x^2+xy+y^2$: $\mathbf{H}=\begin{bmatrix}2&1\\1&2\end{bmatrix}$, $\det=3>0$, $\operatorname{tr}=4>0$: a **minimum**. (Eigenvalues $1$ and $3$.)
> - $f=-x^2+xy-y^2$: $\mathbf{H}=\begin{bmatrix}-2&1\\1&-2\end{bmatrix}$, $\det=3>0$, $\operatorname{tr}=-4<0$: a **maximum**. (Eigenvalues $-1$ and $-3$.)
> - $f=x^2+4xy+y^2$: $\mathbf{H}=\begin{bmatrix}2&4\\4&2\end{bmatrix}$, $\det=4-16=-12<0$: a **saddle**. (Eigenvalues $6$ and $-2$.)
>
> The saddle is worth a second look. Along the diagonal $y=x$, $f=x^2+4x^2+x^2=6x^2$, which goes up. Along $y=-x$, $f=x^2-4x^2+x^2=-2x^2$, which goes down. The two directions are the eigenvectors $(1,1)$ and $(1,-1)$ of $\mathbf{H}$, and the coefficients $6$ and $-2$ are their eigenvalues (Figure 3c). Notice that both diagonal entries of $\mathbf{H}$ are positive: looking only along the axes, $f=x^2$ and $f=y^2$, it seems to be a minimum. The off-diagonal entries matter.

> [!example] Example: one function, four critical points
> Let $f(x,y)=x^3-3x+y^3-3y$, the single-variable example of Figure 1b in each coordinate. Then
>
> $$
> \nabla f=\begin{bmatrix}3x^2-3\\3y^2-3\end{bmatrix},\qquad\mathbf{H}_f=\begin{bmatrix}6x&0\\0&6y\end{bmatrix}.
> $$
>
> The gradient vanishes when $x=\pm1$ and $y=\pm1$, giving four critical points:
>
> | point | $\mathbf{H}_f$ | $\det$ | trace | type | $f$ |
> |---|---|---|---|---|---|
> | $(1,1)$ | $\operatorname{diag}(6,6)$ | $36$ | $12$ | minimum | $-4$ |
> | $(-1,-1)$ | $\operatorname{diag}(-6,-6)$ | $36$ | $-12$ | maximum | $4$ |
> | $(1,-1)$ | $\operatorname{diag}(6,-6)$ | $-36$ | $0$ | saddle | $0$ |
> | $(-1,1)$ | $\operatorname{diag}(-6,6)$ | $-36$ | $0$ | saddle | $0$ |
>
> Figure 3d shows the contours. Around the minimum and maximum they are closed loops; at the saddles they cross, forming an X.

![[Probability and Linear Algebra/images/multivariable-calculus-critical-points.svg]]
*Figure 3. The three kinds of non-degenerate critical point. (a) A minimum: the Hessian is positive definite and the surface is a bowl. (b) A maximum: negative definite, a cap. (c) A saddle, $x^2+4xy+y^2$: it curves up along the eigenvector $(1,1)$ (orange, $6x^2$) and down along $(1,-1)$ (blue, $-2x^2$). (d) The contours of $x^3-3x+y^3-3y$, which has all three kinds: closed loops around the minimum and the maximum, crossing contours at the two saddles.*

### The general case: eigenvalues

With more than two variables, completing the square by hand gets tedious, and the clean statement is in terms of eigenvalues. A symmetric $\mathbf{H}$ can be written $\mathbf{H}=\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ with orthonormal eigenvectors $\mathbf{q}_i$ and real eigenvalues $\lambda_i$, so

$$
\mathbf{h}^\top\mathbf{H}\mathbf{h}=\sum_{i=1}^n\lambda_i\,\big(\mathbf{q}_i^\top\mathbf{h}\big)^2.
$$

Measured along the eigenvector directions, the quadratic form is a weighted sum of squares, with the eigenvalues as weights. Hence:
- all $\lambda_i>0$: minimum; all $\lambda_i<0$: maximum;
- some positive and some negative: saddle (up along the eigenvectors with $\lambda_i>0$, down along those with $\lambda_i<0$);
- some $\lambda_i=0$ and the rest of one sign: inconclusive.

The eigenvalues are the curvatures along the principal directions, and the eigenvectors are the axes of the elliptical contours near a minimum. [[Eigenvalues and Eigenvectors#Positive definite matrices|Eigenvalues and Eigenvectors]] develops positive definite matrices fully and proves the equivalent tests.

> [!example] Example: a 3×3 Hessian
> $f(x,y,z)=x^2+y^2+z^2+xy+yz$ has a critical point at the origin and
>
> $$
> \mathbf{H}=\begin{bmatrix}2&1&0\\1&2&1\\0&1&2\end{bmatrix}.
> $$
>
> Its eigenvalues are $2-\sqrt2\approx0.59$, $2$ and $2+\sqrt2\approx3.41$, all positive, so the origin is a minimum. A quicker hand test is **Sylvester's criterion**: a symmetric matrix is positive definite exactly when all its leading principal minors (the determinants of the top-left $1\times1$, $2\times2$, … blocks) are positive. Here they are $2$, $4-1=3$ and $4$.

### When the test fails

If the Hessian is singular at a critical point, the quadratic model is flat in some direction and higher-order terms decide. Three functions with the **same** Hessian $\operatorname{diag}(2,0)$ at the origin show that anything can happen:
- $f=x^2+y^4$: a minimum (both terms are non-negative);
- $f=x^2-y^4$: a saddle (down along the $y$-axis);
- $f=x^2+y^3$: neither a minimum nor a maximum, and not a saddle in the usual sense either; along the $y$-axis it behaves like $y^3$.

Degenerate critical points are common in machine learning. Overparameterized models have whole valleys of minima, flat in many directions, and their Hessians at a minimum have many zero eigenvalues.

## Matrix calculus

Training a model means differentiating functions of vectors and matrices: $\mathbf{w}^\top\mathbf{x}$, $\lVert\mathbf{X}\mathbf{w}-\mathbf{y}\rVert^2$, $\log\det\boldsymbol\Sigma$. Doing this one component at a time works, but it is slow and error-prone. Matrix calculus is the bookkeeping that lets us do it in one line.[^parr2018]

### Layout convention

Conventions differ between books, so here is the one used throughout the series:
- the **gradient** of a scalar with respect to a vector $\mathbf{x}\in\mathbb{R}^n$ is a **column vector** in $\mathbb{R}^n$, the same shape as $\mathbf{x}$;
- the **Jacobian** of $\mathbf{F}:\mathbb{R}^n\to\mathbb{R}^m$ is $m\times n$, one row per output, so $\nabla f=\mathbf{J}_f^\top$ for scalar $f$;
- the gradient of a scalar with respect to a matrix $\mathbf{X}\in\mathbb{R}^{m\times n}$ is an $m\times n$ matrix, the same shape as $\mathbf{X}$, with entries $\big(\nabla_\mathbf{X}f\big)_{ij}=\partial f/\partial X_{ij}$.

The "same shape as the variable" rule is what makes the update $\mathbf{x}\leftarrow\mathbf{x}-\alpha\nabla f$ type-check, and it is what PyTorch and JAX return.

### The perturbation method

There are two ways to compute a gradient. One is **components**: write $f$ as a sum over indices and take $\partial/\partial x_k$. The other, usually faster, is the **perturbation method**, which uses definition (3) directly: expand $f(\mathbf{x}+\mathbf{h})$, keep the terms that are linear in $\mathbf{h}$, and write them as $\mathbf{g}^\top\mathbf{h}$. Then $\mathbf{g}=\nabla f(\mathbf{x})$. For a matrix variable, write the linear term as $\operatorname{tr}(\mathbf{G}^\top\mathbf{H})=\sum_{ij}G_{ij}H_{ij}$, and then $\mathbf{G}=\nabla_\mathbf{X}f$. We use both methods below, so that each result is derived twice.

### Linear functions: $\nabla(\mathbf{a}^\top\mathbf{x})=\mathbf{a}$

$f(\mathbf{x})=\mathbf{a}^\top\mathbf{x}=\sum_ia_ix_i$, so $\partial f/\partial x_k=a_k$ and $\nabla f=\mathbf{a}$. With perturbations: $\mathbf{a}^\top(\mathbf{x}+\mathbf{h})=\mathbf{a}^\top\mathbf{x}+\mathbf{a}^\top\mathbf{h}$, and the linear term is $\mathbf{a}^\top\mathbf{h}$. For example, $f=2x_1-x_2+3x_3$ has gradient $(2,-1,3)$ everywhere. This is the vector version of $\frac{d}{dx}(ax)=a$.

### Quadratic forms: $\nabla(\mathbf{x}^\top\mathbf{A}\mathbf{x})=(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}$

**Components.** $f(\mathbf{x})=\mathbf{x}^\top\mathbf{A}\mathbf{x}=\sum_{i,j}A_{ij}x_ix_j$. The variable $x_k$ appears in the terms with $i=k$ and in those with $j=k$, so

$$
\frac{\partial f}{\partial x_k}=\sum_jA_{kj}x_j+\sum_iA_{ik}x_i=(\mathbf{A}\mathbf{x})_k+(\mathbf{A}^\top\mathbf{x})_k.
$$

**Perturbations.** Expanding,

$$
\begin{aligned}
(\mathbf{x}+\mathbf{h})^\top\mathbf{A}(\mathbf{x}+\mathbf{h})&=\mathbf{x}^\top\mathbf{A}\mathbf{x}+\mathbf{h}^\top\mathbf{A}\mathbf{x}+\mathbf{x}^\top\mathbf{A}\mathbf{h}+\mathbf{h}^\top\mathbf{A}\mathbf{h}\\
&=\mathbf{x}^\top\mathbf{A}\mathbf{x}+\big((\mathbf{A}+\mathbf{A}^\top)\mathbf{x}\big)^\top\mathbf{h}+O(\lVert\mathbf{h}\rVert^2),
\end{aligned}
$$

using $\mathbf{h}^\top\mathbf{A}\mathbf{x}=\mathbf{x}^\top\mathbf{A}^\top\mathbf{h}$ (a number equals its own transpose). Either way,

$$
\nabla\big(\mathbf{x}^\top\mathbf{A}\mathbf{x}\big)=(\mathbf{A}+\mathbf{A}^\top)\,\mathbf{x},\qquad\nabla^2\big(\mathbf{x}^\top\mathbf{A}\mathbf{x}\big)=\mathbf{A}+\mathbf{A}^\top.
$$

For **symmetric** $\mathbf{A}$ this is $2\mathbf{A}\mathbf{x}$, the vector version of $\frac{d}{dx}ax^2=2ax$.

> [!example] Example: checking by hand
> Take $\mathbf{A}=\begin{bmatrix}1&2\\0&3\end{bmatrix}$, which is not symmetric. Then $\mathbf{x}^\top\mathbf{A}\mathbf{x}=x_1^2+2x_1x_2+3x_2^2$, whose gradient is $(2x_1+2x_2,\ 2x_1+6x_2)$. The formula gives $\mathbf{A}+\mathbf{A}^\top=\begin{bmatrix}2&2\\2&6\end{bmatrix}$, and $\begin{bmatrix}2&2\\2&6\end{bmatrix}\mathbf{x}$ is the same vector. Writing $2\mathbf{A}\mathbf{x}$ here would give $(2x_1+4x_2,\ 6x_2)$: wrong. Only the symmetric part of $\mathbf{A}$ matters to a quadratic form.

### Least squares: $\nabla\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2$ and the normal equations

Let $f(\mathbf{x})=\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2$ with $\mathbf{A}\in\mathbb{R}^{m\times n}$, and write $\mathbf{r}=\mathbf{A}\mathbf{x}-\mathbf{b}$ for the residual. Perturbing $\mathbf{x}$ changes the residual by $\mathbf{A}\mathbf{h}$:

$$
\begin{aligned}
\lVert\mathbf{r}+\mathbf{A}\mathbf{h}\rVert^2&=\lVert\mathbf{r}\rVert^2+2\,\mathbf{r}^\top\mathbf{A}\mathbf{h}+\lVert\mathbf{A}\mathbf{h}\rVert^2\\
&=\lVert\mathbf{r}\rVert^2+\big(2\mathbf{A}^\top\mathbf{r}\big)^\top\mathbf{h}+O(\lVert\mathbf{h}\rVert^2).
\end{aligned}
$$

So

$$
\nabla f(\mathbf{x})=2\mathbf{A}^\top(\mathbf{A}\mathbf{x}-\mathbf{b}),\qquad\nabla^2f=2\mathbf{A}^\top\mathbf{A}.
$$

(The same result follows from the quadratic-form rule: $f=\mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}-2\mathbf{b}^\top\mathbf{A}\mathbf{x}+\mathbf{b}^\top\mathbf{b}$.) Setting the gradient to zero gives the **normal equations**

$$
\mathbf{A}^\top\mathbf{A}\,\mathbf{x}=\mathbf{A}^\top\mathbf{b}.
\tag{7}
$$

The Hessian $2\mathbf{A}^\top\mathbf{A}$ is positive semidefinite, because $\mathbf{h}^\top\mathbf{A}^\top\mathbf{A}\mathbf{h}=\lVert\mathbf{A}\mathbf{h}\rVert^2\ge0$, so (as the [[#Convexity|convexity]] section will show) every solution of (7) is a global minimum. The condition $\mathbf{A}^\top\mathbf{r}=\mathbf{0}$ says the residual is perpendicular to every column of $\mathbf{A}$: the geometric version of the same equations, developed in [[Orthogonality and Least Squares#Least squares|Orthogonality and Least Squares]]. In regression, $\mathbf{A}$ is the data matrix $\mathbf{X}$, $\mathbf{x}$ is the weight vector $\mathbf{w}$ and $\mathbf{b}$ is the target $\mathbf{y}$; see [[Linear Models#Solving it: the normal equations|Linear Models]].

> [!example] Example: three equations, two unknowns
> Solve $x_1=1$, $x_2=1$, $x_1+x_2=0$ as well as possible. These are inconsistent. In matrix form,
>
> $$
> \mathbf{A}=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad\mathbf{b}=\begin{bmatrix}1\\1\\0\end{bmatrix},
> $$
>
> $$
> \mathbf{A}^\top\mathbf{A}=\begin{bmatrix}2&1\\1&2\end{bmatrix},\quad\mathbf{A}^\top\mathbf{b}=\begin{bmatrix}1\\1\end{bmatrix}.
> $$
>
> The normal equations $2x_1+x_2=1$, $x_1+2x_2=1$ give $x_1=x_2=\frac13$. The residual is $\mathbf{r}=(-\frac23,-\frac23,\frac23)$, and indeed $\mathbf{A}^\top\mathbf{r}=(-\frac23+\frac23,\ -\frac23+\frac23)=\mathbf{0}$. The compromise splits the disagreement evenly between the three equations.

### Log-sum-exp and the softmax

The **log-sum-exp** function $\operatorname{lse}(\mathbf{z})=\log\sum_{j=1}^Ke^{z_j}$ is a smooth version of $\max_jz_j$, and it is the normalizer of every softmax classifier. By the chain rule,

$$
\frac{\partial}{\partial z_k}\log\sum_je^{z_j}=\frac{e^{z_k}}{\sum_je^{z_j}},
$$

so

$$
\nabla\operatorname{lse}(\mathbf{z})=\operatorname{softmax}(\mathbf{z})=:\mathbf{p}.
$$

The gradient of log-sum-exp is the softmax. Differentiating $p_k=e^{z_k}/\sum_je^{z_j}$ once more (quotient rule) gives $\partial p_k/\partial z_j=p_k(\delta_{kj}-p_j)$, where $\delta_{kj}$ is 1 if $k=j$ and 0 otherwise. So the Hessian is

$$
\nabla^2\operatorname{lse}(\mathbf{z})=\operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top.
$$

This matrix is positive semidefinite, and the reason is pretty: for any $\mathbf{v}$,

$$
\mathbf{v}^\top\big(\operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top\big)\mathbf{v}=\sum_kp_kv_k^2-\Big(\sum_kp_kv_k\Big)^2,
$$

which is the variance of a random variable that takes the value $v_k$ with probability $p_k$. Variances are never negative, so log-sum-exp is convex. The variance is zero for $\mathbf{v}=\mathbf{1}$: adding the same constant to every score changes nothing, which is why the softmax is unchanged by shifts and why we subtract $\max_jz_j$ before exponentiating in code.

> [!example] Example: scores $\mathbf{z}=(0,\ \ln2,\ \ln3)$
> Then $e^{\mathbf{z}}=(1,2,3)$, which sums to $6$. So $\operatorname{lse}(\mathbf{z})=\ln6\approx1.79$, slightly above $\max_jz_j=\ln3\approx1.10$, and $\nabla\operatorname{lse}=\mathbf{p}=(\frac16,\frac13,\frac12)$. For the cross-entropy loss of class $y$, $\ell(\mathbf{z})=-z_y+\operatorname{lse}(\mathbf{z})$, the gradient is $\mathbf{p}-\mathbf{e}_y$: "predicted minus true". If the true class is the third, it is $(\frac16,\frac13,-\frac12)$, and gradient descent raises $z_3$ and lowers the others. This is the gradient of [[Linear Models#More than two classes: softmax regression|softmax regression]].

### Matrix variables: the trace and the log-determinant

Two more identities are needed for Gaussian models, briefly.

**The trace.** $\operatorname{tr}(\mathbf{A}\mathbf{X})=\sum_i(\mathbf{A}\mathbf{X})_{ii}=\sum_{i,j}A_{ij}X_{ji}$. The entry $X_{ji}$ appears once, with coefficient $A_{ij}$, so $\partial/\partial X_{ji}=A_{ij}$, and arranging these in the shape of $\mathbf{X}$:

$$
\nabla_\mathbf{X}\operatorname{tr}(\mathbf{A}\mathbf{X})=\mathbf{A}^\top.
$$

**The log-determinant.** For invertible $\mathbf{X}$ with $\det\mathbf{X}>0$, factor $\mathbf{X}+\mathbf{H}=\mathbf{X}(\mathbf{I}+\mathbf{X}^{-1}\mathbf{H})$. For a small matrix $\mathbf{E}$, $\det(\mathbf{I}+\mathbf{E})=1+\operatorname{tr}\mathbf{E}+O(\lVert\mathbf{E}\rVert^2)$, because the only first-order terms in the expansion of the determinant come from the diagonal. Hence

$$
\begin{aligned}
\log\det(\mathbf{X}+\mathbf{H})&=\log\det\mathbf{X}+\log\big(1+\operatorname{tr}(\mathbf{X}^{-1}\mathbf{H})+\cdots\big)\\
&=\log\det\mathbf{X}+\operatorname{tr}(\mathbf{X}^{-1}\mathbf{H})+\cdots
\end{aligned}
$$

and reading off the linear term in the form $\operatorname{tr}(\mathbf{G}^\top\mathbf{H})$,

$$
\nabla_\mathbf{X}\log\det\mathbf{X}=\mathbf{X}^{-\top},
$$

which is $\mathbf{X}^{-1}$ for symmetric $\mathbf{X}$, such as a covariance matrix. In one dimension this is $\frac{d}{dx}\log x=\frac1x$. For example, $\mathbf{X}=\begin{bmatrix}2&1\\1&1\end{bmatrix}$ has $\det\mathbf{X}=1$ and gradient $\mathbf{X}^{-1}=\begin{bmatrix}1&-1\\-1&2\end{bmatrix}$. Together with the trace rule, this gives the maximum-likelihood covariance of a Gaussian: setting the gradient of the log-likelihood with respect to $\boldsymbol\Sigma^{-1}$ to zero yields the sample covariance.

### A table to keep

| $f$ | $\nabla f$ | Hessian |
|---|---|---|
| $\mathbf{a}^\top\mathbf{x}$ | $\mathbf{a}$ | $\mathbf{0}$ |
| $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ | $(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}$ | $\mathbf{A}+\mathbf{A}^\top$ |
| $\lVert\mathbf{x}\rVert^2$ | $2\mathbf{x}$ | $2\mathbf{I}$ |
| $\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2$ | $2\mathbf{A}^\top(\mathbf{A}\mathbf{x}-\mathbf{b})$ | $2\mathbf{A}^\top\mathbf{A}$ |
| $\operatorname{lse}(\mathbf{z})$ | $\operatorname{softmax}(\mathbf{z})=\mathbf{p}$ | $\operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$ |
| $\operatorname{tr}(\mathbf{A}\mathbf{X})$ | $\mathbf{A}^\top$ | |
| $\log\det\mathbf{X}$ | $\mathbf{X}^{-\top}$ | |

The *Matrix Cookbook* collects hundreds more.[^cookbook]

### Checking gradients numerically

A gradient derived by hand can be checked with the **central difference** $\big(f(\mathbf{x}+\epsilon\mathbf{e}_i)-f(\mathbf{x}-\epsilon\mathbf{e}_i)\big)/(2\epsilon)$ for each coordinate $i$. By (2), its error is $O(\epsilon^2)$, because the second-order terms cancel between the two sides; the one-sided difference has error $O(\epsilon)$. With $\epsilon\approx10^{-6}$ in double precision, a correct gradient agrees to a relative error of about $10^{-10}$, while a wrong one is off by percents. [[Neural Networks#Gradient checking|Neural Networks]] uses the same check for backpropagation.

> [!example]- Code: a numerical gradient check for the identities above
> ```python
> import numpy as np
> rng = np.random.default_rng(0)
>
> def grad_check(f, grad, x, eps=1e-6):
>     """Relative error between an analytic gradient and central differences."""
>     g_num = np.zeros_like(x)
>     for i in range(x.size):
>         e = np.zeros_like(x); e.flat[i] = eps
>         g_num.flat[i] = (f(x + e) - f(x - e)) / (2 * eps)
>     g = grad(x)
>     return np.linalg.norm(g - g_num) / (np.linalg.norm(g) + np.linalg.norm(g_num))
>
> A = rng.standard_normal((5, 3)); b = rng.standard_normal(5); M = rng.standard_normal((3, 3))
> x = rng.standard_normal(3)
> print(grad_check(lambda x: x @ M @ x, lambda x: (M + M.T) @ x, x))                      # quadratic form
> print(grad_check(lambda x: np.sum((A @ x - b)**2), lambda x: 2 * A.T @ (A @ x - b), x))  # least squares
> softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
> lse = lambda z: z.max() + np.log(np.exp(z - z.max()).sum())
> print(grad_check(lse, softmax, x))                                                      # log-sum-exp
> print(grad_check(lambda x: x @ M @ x, lambda x: 2 * M @ x, x))                          # a classic bug
> ```
> The first three print about $2\times10^{-11}$. The last uses $2\mathbf{M}\mathbf{x}$ for a non-symmetric $\mathbf{M}$ and prints $0.27$: the check catches the bug at once.

## Convexity

Minimizing a general function is hard: gradient descent can get stuck in a local minimum, and a zero gradient might be a saddle. For **convex** functions these problems disappear. That is why so much of classical machine learning (least squares, logistic regression, SVMs) is built on convex losses, and why the analysis of [[First-Order Optimization]] starts there.[^boyd2004]

### Convex sets

A set $C\subseteq\mathbb{R}^n$ is **convex** if the line segment between any two of its points stays inside it: for all $\mathbf{x},\mathbf{y}\in C$ and $\theta\in[0,1]$, the point $\theta\mathbf{x}+(1-\theta)\mathbf{y}$ is in $C$.
- **Convex:** a half-space $\{\mathbf{x}:\mathbf{a}^\top\mathbf{x}\le b\}$, a ball $\{\lVert\mathbf{x}\rVert\le r\}$, a line, the whole space, the set of probability vectors. Intersections of convex sets are convex, so a polyhedron $\{\mathbf{A}\mathbf{x}\le\mathbf{b}\}$ is convex.
- **Not convex:** a ring (annulus), the union of two separate disks, the circle $\{\lVert\mathbf{x}\rVert=1\}$ itself (the segment between two points on it cuts through the inside), and the set $\{x:\ x^2\ge1\}$.

### Convex functions

A function $f$ on a convex set is **convex** if its graph lies below every chord:

$$
f\big(\theta\mathbf{x}+(1-\theta)\mathbf{y}\big)\le\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})
$$

for all $\mathbf{x},\mathbf{y}$ in the set and all $\theta\in[0,1]$.
The left side is the function at a point between $\mathbf{x}$ and $\mathbf{y}$; the right side is the chord at the same point (Figure 4a). Equivalently, the region above the graph is a convex set. A function is **concave** if $-f$ is convex.

> [!example] Example: $x^2$ is convex, straight from the definition
> Subtract the two sides:
>
> $$
> \theta x^2+(1-\theta)y^2-\big(\theta x+(1-\theta)y\big)^2=\theta(1-\theta)(x-y)^2\ge0.
> $$
>
> (Expand the square and collect terms; $\theta-\theta^2=\theta(1-\theta)$.) The gap between chord and graph is largest in the middle of the segment and grows with the square of the distance between the points.

The definition is rarely the easiest way to check convexity. For differentiable functions there are two better tests.

### The first-order condition: tangents lie below the graph

A differentiable $f$ is convex if and only if

$$
f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})\qquad\text{for all }\mathbf{x},\mathbf{y}.
\tag{8}
$$

The right side is the tangent plane at $\mathbf{x}$, evaluated at $\mathbf{y}$. So a convex function lies **above all its tangent planes** (Figure 4a). This is a remarkable statement: purely local information at $\mathbf{x}$, the value and the gradient, gives a **global** lower bound on $f$ everywhere.

> [!proof]- Proof of the first-order condition
> ($\Rightarrow$) Convexity with $\theta\in(0,1]$, rearranged, says $f\big(\mathbf{x}+\theta(\mathbf{y}-\mathbf{x})\big)-f(\mathbf{x})\le\theta\big(f(\mathbf{y})-f(\mathbf{x})\big)$. Divide by $\theta$ and let $\theta\to0$. The left side tends to the directional derivative $\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$, by (4), which gives (8).
>
> ($\Leftarrow$) Let $\mathbf{z}=\theta\mathbf{x}+(1-\theta)\mathbf{y}$. Apply (8) at $\mathbf{z}$ twice, towards $\mathbf{x}$ and towards $\mathbf{y}$:
> $f(\mathbf{x})\ge f(\mathbf{z})+\nabla f(\mathbf{z})^\top(\mathbf{x}-\mathbf{z})$ and $f(\mathbf{y})\ge f(\mathbf{z})+\nabla f(\mathbf{z})^\top(\mathbf{y}-\mathbf{z})$.
> Multiply the first by $\theta$, the second by $1-\theta$, and add. The gradient terms cancel, because $\theta(\mathbf{x}-\mathbf{z})+(1-\theta)(\mathbf{y}-\mathbf{z})=\mathbf{0}$, leaving $\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})\ge f(\mathbf{z})$.

The most important consequence: if $\nabla f(\mathbf{x}^*)=\mathbf{0}$, then (8) says $f(\mathbf{y})\ge f(\mathbf{x}^*)$ for **every** $\mathbf{y}$. For a differentiable convex function,

$$
\nabla f(\mathbf{x}^*)=\mathbf{0}\quad\Longleftrightarrow\quad\mathbf{x}^*\text{ is a global minimizer.}
$$

The gradient test, only a necessary condition in general, becomes necessary **and sufficient**. No saddles, no local traps.

### The second-order condition: the Hessian is positive semidefinite

A twice-differentiable $f$ on an open convex set is convex if and only if its Hessian is positive semidefinite everywhere:

$$
\mathbf{H}_f(\mathbf{x})\succeq0\qquad\text{for all }\mathbf{x},
$$

meaning $\mathbf{h}^\top\mathbf{H}_f(\mathbf{x})\mathbf{h}\ge0$ for every $\mathbf{h}$. In one variable this is $f''\ge0$: the slope never decreases, so the graph never bends down. In many variables it says that the function curves upward (or is flat) in every direction, at every point. The reason is the reduction we used for (6). Convexity of $f$ is the same as convexity of every one-dimensional slice $g(t)=f(\mathbf{x}+t\mathbf{h})$, and $g''(t)=\mathbf{h}^\top\mathbf{H}_f\mathbf{h}$.

This is usually the quickest test. Each example we have met:
- $x^2+2y^2$: Hessian $\operatorname{diag}(2,4)\succ0$, convex;
- $\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2$: Hessian $2\mathbf{A}^\top\mathbf{A}\succeq0$, convex;
- $\operatorname{lse}$: Hessian $\operatorname{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top\succeq0$ (a variance), convex;
- $f(x,y)=xy$: Hessian $\begin{bmatrix}0&1\\1&0\end{bmatrix}$, with eigenvalues $\pm1$; indefinite, so **not** convex. It is the saddle $x^2+4xy+y^2$ in disguise.

### Every local minimum is global

This is the property that makes convex problems tractable, and it needs no differentiability.

> [!proof]- Proof that a local minimum of a convex function is global
> Suppose $\mathbf{x}^*$ is a local minimum, so $f(\mathbf{x}^*)\le f(\mathbf{z})$ for all $\mathbf{z}$ within some distance $\delta$ of $\mathbf{x}^*$, but that some $\mathbf{y}$ has $f(\mathbf{y})<f(\mathbf{x}^*)$. Walk from $\mathbf{x}^*$ a little way towards $\mathbf{y}$: $\mathbf{z}=\theta\mathbf{y}+(1-\theta)\mathbf{x}^*$ with $\theta>0$ small enough that $\mathbf{z}$ is within $\delta$. By convexity,
>
> $$
> \begin{aligned}
> f(\mathbf{z})&\le\theta f(\mathbf{y})+(1-\theta)f(\mathbf{x}^*)\\
> &<\theta f(\mathbf{x}^*)+(1-\theta)f(\mathbf{x}^*)=f(\mathbf{x}^*),
> \end{aligned}
> $$
>
> so $\mathbf{z}$ is a nearby point with a smaller value, contradicting local minimality. Hence no such $\mathbf{y}$ exists.

Figure 4b shows what goes wrong without convexity. The function $x^4-3x^2+x$ has a global minimum near $x=-1.30$ (value $-3.51$) and a second, merely local minimum near $x=1.13$ (value $-1.07$). The chord between them passes **below** the hump in between, which is exactly the violation of the convexity inequality. Gradient descent started on the right converges to the worse minimum.

Two more facts, both from the same kind of argument: the set of global minimizers of a convex function is itself convex (it can be a single point, a segment, a whole flat region, or empty), and the maximum of a convex function over a convex polytope is attained at a corner.

![[Probability and Linear Algebra/images/multivariable-calculus-convexity.svg]]
*Figure 4. (a) A convex function lies below each of its chords (red) and above each of its tangent lines (blue). (b) $x^4-3x^2+x$ is not convex: the chord between its two minima dips below the graph. Its local minimum at $x\approx1.13$ is not the global one at $x\approx-1.30$, and a zero derivative no longer certifies optimality.*

### Strict and strong convexity

Convexity allows flat stretches, and with them many minimizers or none at all. Two stronger notions fix this.
- **Strictly convex**: the chord inequality is strict ($<$) whenever $\mathbf{x}\ne\mathbf{y}$ and $\theta\in(0,1)$. A strictly convex function has **at most one** minimizer. A positive definite Hessian everywhere is sufficient but not necessary: $x^4$ is strictly convex although $f''(0)=0$. Strict convexity does not guarantee that a minimizer exists: $e^x$ is strictly convex and has none.
- **Strongly convex** with parameter $m>0$: $f(\mathbf{x})-\frac m2\lVert\mathbf{x}\rVert^2$ is still convex. For twice-differentiable $f$ this means $\mathbf{H}_f\succeq m\mathbf{I}$: the curvature is at least $m$ in every direction, everywhere. Equivalently, (8) improves to $f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})+\frac m2\lVert\mathbf{y}-\mathbf{x}\rVert^2$: a quadratic lower bound instead of a linear one. A strongly convex function has **exactly one** minimizer, and it grows at least quadratically away from it.

$x^2$ is strongly convex with $m=2$, while $x^4$ and $e^x$ are strictly but not strongly convex (their curvature gets arbitrarily close to zero). Adding an $\ell_2$ penalty makes any convex loss strongly convex: ridge regression, $\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2+\lambda\lVert\mathbf{x}\rVert^2$, has Hessian $2\mathbf{A}^\top\mathbf{A}+2\lambda\mathbf{I}\succeq2\lambda\mathbf{I}$, so it always has a unique solution. Strong convexity is what turns the slow $O(1/t)$ convergence of gradient descent into a fast, geometric one ([[First-Order Optimization#Convergence rates for smooth and convex functions|First-Order Optimization]]).

### A catalogue of examples

**Convex:**
- affine functions $\mathbf{a}^\top\mathbf{x}+b$ (both convex and concave);
- $x^2$, $\lvert x\rvert$, $e^{ax}$, $-\log x$ and $x\log x$ (for $x>0$), and every norm $\lVert\mathbf{x}\rVert$;
- $\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2$, log-sum-exp, the logistic loss $\log(1+e^{-z})$ and the hinge loss $\max(0,1-z)$;
- anything built from convex functions by **non-negative sums**, **composition with an affine map** ($f(\mathbf{A}\mathbf{x}+\mathbf{b})$), and **pointwise maxima**. These rules show at once that the losses of [[Linear Models]] and [[Support Vector Machines]] are convex in the weights.

**Not convex:**
- $x^3$ (bends down for $x<0$), $\sin x$, $x^4-3x^2+x$;
- $xy$ (a saddle), and $\sqrt x$ and $\log x$ (concave);
- the loss of a neural network as a function of its weights. Swapping two hidden units gives a different weight vector with the same loss, and the average of the two is usually worse, which a convex function forbids.

## Constrained optimization: Lagrange multipliers

So far the variables were free. Often they are not: probabilities must sum to one, a vector must have unit length, a margin must be at least one. This section starts with equality constraints and their geometry, then turns inequality constraints back into unconstrained problems, first with penalties and then with the Lagrangian, and ends with the KKT conditions that the [[Support Vector Machines|SVM]] is built on.

### Equality constraints

The problem is to maximize (or minimize) $f(\mathbf{x})$ subject to $g(\mathbf{x})=c$: we may only move along the **constraint set** $\{g=c\}$, which in two dimensions is a curve.

**The geometry.** Suppose $\mathbf{x}^*$ is a constrained maximum. Walking along the constraint curve through $\mathbf{x}^*$ cannot increase $f$ to first order, so the directional derivative of $f$ along the curve is zero: $\nabla f(\mathbf{x}^*)$ is perpendicular to the curve's tangent. But the constraint curve is a level set of $g$, and by the result of [[#The gradient is perpendicular to the level sets]], $\nabla g(\mathbf{x}^*)$ is perpendicular to it too. Two vectors perpendicular to the same line in the plane are parallel, so

$$
\nabla f(\mathbf{x}^*)=\lambda\,\nabla g(\mathbf{x}^*)
\tag{9}
$$

for some number $\lambda$, the **Lagrange multiplier**. In words: at a constrained optimum, the level set of $f$ **touches** the constraint curve tangentially. If they crossed instead, we could slide along the constraint into a higher level of $f$ (Figure 5). The same argument works in $n$ dimensions, where the constraint set is an $(n-1)$-dimensional surface and "perpendicular to every tangent direction" pins $\nabla f$ to the line of $\nabla g$.

**The recipe.** Condition (9) together with the constraint gives $n+1$ equations in the $n+1$ unknowns $\mathbf{x}$ and $\lambda$. They are packaged as the stationarity conditions of the **Lagrangian**

$$
\mathcal{L}(\mathbf{x},\lambda)=f(\mathbf{x})-\lambda\big(g(\mathbf{x})-c\big).
$$

Setting $\nabla_\mathbf{x}\mathcal{L}=\mathbf{0}$ gives (9), and setting $\partial\mathcal{L}/\partial\lambda=0$ gives back the constraint.

Solve, list all candidate points, and compare the values of $f$. Two caveats. The argument needs $\nabla g(\mathbf{x}^*)\ne\mathbf{0}$ (a **constraint qualification**: otherwise the constraint set may have a corner, and (9) can fail). And (9) is only necessary: it finds maxima, minima and saddles of $f$ on the constraint set alike. If the constraint set is closed and bounded, $f$ attains a maximum and a minimum on it, so comparing values among the candidates settles the question.

> [!example] Example 1: maximize $x+y$ on the unit circle
> Here $f=x+y$ and $g=x^2+y^2=1$. Condition (9) reads $(1,1)=\lambda(2x,2y)$, so $x=y=\frac{1}{2\lambda}$. The constraint gives $2x^2=1$, so $x=y=\pm\frac{1}{\sqrt2}$. The two candidates are $(\frac1{\sqrt2},\frac1{\sqrt2})$ with $f=\sqrt2$, the maximum, and $(-\frac1{\sqrt2},-\frac1{\sqrt2})$ with $f=-\sqrt2$, the minimum. At the maximum $\lambda=\frac{1}{\sqrt2}$. Figure 5 shows the tangency: the line $x+y=\sqrt2$ just touches the circle.

> [!example] Example 2: the largest box with a given surface area
> A closed box has sides $x,y,z>0$ and surface area $2(xy+yz+zx)=S$. Maximize its volume $V=xyz$. Condition (9) gives three equations,
>
> $$
> yz=2\lambda(y+z),\qquad xz=2\lambda(x+z),\qquad xy=2\lambda(x+y).
> $$
>
> Multiply the first by $x$ and the second by $y$: both left sides become $xyz$, so $2\lambda(xy+xz)=2\lambda(xy+yz)$, hence $xz=yz$ and $x=y$ (as $z>0$, and $\lambda\ne0$ since otherwise $yz=0$). The same trick with the second and third gives $y=z$. The best box is a **cube**. With $S=24$: $6x^2=24$, so $x=2$, $V=8$, and $\lambda=\frac{yz}{2(y+z)}=\frac{4}{8}=\frac12$. It is a maximum because $V\to0$ as the box flattens out towards the edges of the allowed region, so the single interior candidate must be the maximum.

> [!example] Example 3: the maximum-entropy distribution
> Which distribution $\mathbf{p}=(p_1,\dots,p_n)$ on $n$ outcomes has the largest entropy $H(\mathbf{p})=-\sum_ip_i\log p_i$? The constraint is $\sum_ip_i=1$, with gradient $\mathbf{1}$. Since $\partial H/\partial p_i=-\log p_i-1$, condition (9) says
>
> $$
> -\log p_i-1=\lambda\quad\text{for every }i,
> $$
>
> so all $p_i$ are equal: $p_i=\frac1n$, the **uniform** distribution, with $H=\log n$. (We ignored $p_i\ge0$; the answer is strictly positive, so that constraint is inactive. $H$ is concave, so the stationary point is the global maximum.) With no information beyond normalization, the least committed distribution spreads out evenly.
>
> Now add a second constraint, a known mean $\sum_ii\,p_i=m$. With two constraints, (9) becomes $\nabla f=\lambda_1\nabla g_1+\lambda_2\nabla g_2$, here $-\log p_i-1=\lambda_1+\lambda_2\,i$, so
>
> $$
> p_i=\frac{e^{\beta i}}{\sum_je^{\beta j}}
> $$
>
> for some $\beta$ (equal to $-\lambda_2$) chosen to match the mean. This is the **Boltzmann distribution**, a softmax of the scores $\beta i$. For a six-sided die known to average $4.5$ instead of $3.5$, solving numerically gives $\beta\approx0.371$ and $\mathbf{p}\approx(0.054,0.079,0.114,0.165,0.240,0.348)$: probabilities that tilt smoothly towards the high faces. Jaynes made this the **principle of maximum entropy**,[^jaynes1957] and it is one way to see why softmax and Boltzmann distributions appear everywhere ([[Energy-Based Models#The Boltzmann distribution|Energy-Based Models]]).

> [!example] Example 4: the direction of maximum variance
> Maximize $\mathbf{w}^\top\mathbf{S}\mathbf{w}$ subject to $\mathbf{w}^\top\mathbf{w}=1$, for a symmetric $\mathbf{S}$ such as a covariance matrix. The gradients are $2\mathbf{S}\mathbf{w}$ and $2\mathbf{w}$, so (9) reads $\mathbf{S}\mathbf{w}=\lambda\mathbf{w}$: the candidates are the **eigenvectors** of $\mathbf{S}$, and the objective there equals $\mathbf{w}^\top\mathbf{S}\mathbf{w}=\lambda$. The maximum is the largest eigenvalue, attained at its eigenvector. This one line is the derivation of [[Matrix Factorization and Dimensionality Reduction#Principal component analysis (PCA)|principal component analysis]] and of the [[Eigenvalues and Eigenvectors#Trace, determinant and the Rayleigh quotient|Rayleigh quotient]] bounds.

**What the multiplier means.** Let $f^*(c)$ be the optimal value as a function of the constraint level $c$. Then

$$
\frac{df^*}{dc}=\lambda.
$$

The multiplier is the **price** of the constraint: how much the optimum improves per unit of relaxation. In Example 1 with the circle $x^2+y^2=c$, the maximum is $f^*(c)=\sqrt{2c}$, whose derivative at $c=1$ is $\frac{1}{\sqrt2}=\lambda$. In Example 2, $V^*(S)=(S/6)^{3/2}$, whose derivative at $S=24$ is $\frac32\cdot2\cdot\frac16=\frac12=\lambda$: one extra unit of cardboard buys half a unit of volume. (The reason, briefly: at the optimum $\mathcal{L}=f^*$, and by the chain rule only the explicit dependence of $\mathcal{L}$ on $c$, which is $+\lambda$, survives.)

![[Probability and Linear Algebra/images/multivariable-calculus-lagrange.svg]]
*Figure 5. Maximizing $f=x+y$ on the unit circle. The grey lines are level sets of $f$. At the maximum $(\frac{1}{\sqrt2},\frac{1}{\sqrt2})$ the level set $x+y=\sqrt2$ (red) is tangent to the circle, and $\nabla f$ and $\nabla g$ point the same way. At a non-optimal point (orange) they are not parallel: $\nabla f$ has a component along the circle's tangent, so sliding along the circle raises $f$.*

> [!example]- Code: checking Example 2 and the KKT example numerically
> ```python
> import numpy as np
> from scipy.optimize import minimize
> # Example 2: maximize xyz subject to 2(xy + yz + zx) = 24
> cons = [{"type": "eq", "fun": lambda v: 2 * (v[0]*v[1] + v[1]*v[2] + v[2]*v[0]) - 24}]
> res = minimize(lambda v: -np.prod(v), x0=[1, 2, 3], constraints=cons, method="SLSQP")
> print(res.x.round(4), -res.fun)                    # [2. 2. 2.] 8.0
> # KKT example below: minimize (x-2)^2 + (y-1)^2 subject to x + y <= 1
> cons = [{"type": "ineq", "fun": lambda v: 1 - v[0] - v[1]}]   # scipy wants fun(v) >= 0
> res = minimize(lambda v: (v[0] - 2)**2 + (v[1] - 1)**2, x0=[0, 0], constraints=cons, method="SLSQP")
> print(res.x.round(4), res.fun)                     # [1. 0.] 2.0
> ```

### Inequality constraints and penalties

Most constraints in machine learning are inequalities. Write them in the standard form

$$
\min_\mathbf{x}\ f(\mathbf{x})\qquad\text{subject to}\qquad g_i(\mathbf{x})\le0,\quad i=1,\dots,m.
$$

**Unconstrained problems are easy to characterize.** For differentiable $f$, every local minimizer satisfies $\nabla f(\mathbf{x}^*)=\mathbf{0}$, and for convex $f$ the converse holds too ([[#Convexity]]).

**Constrained problems are not.** With constraints, $\nabla f=\mathbf{0}$ is **neither necessary nor sufficient**. Take the toy problem

$$
\min_x\ x^2\qquad\text{subject to}\qquad x\ge1,\quad\text{that is,}\quad g(x)=1-x\le0.
$$

The optimum is $x^*=1$, where $f'(1)=2\ne0$. The only point with $f'=0$ is $x=0$, which is infeasible. The optimality test itself has to change. Notice there are two possible situations at a solution: either the constraint is **inactive** ($g<0$, the solution is in the interior and the ordinary test $\nabla f=\mathbf{0}$ applies), or it is **active** ($g=0$, the solution is on the boundary and behaves like an equality constraint, as in the previous section). The rest of this section builds a single test that covers both, by turning the constrained problem back into an unconstrained one.

**Idea 1: penalize infeasibility.** Replace the constraints by a penalty $P(\mathbf{x})$ that is zero on the feasible set and positive outside it, and minimize $f+P$ with no constraints. The **ideal** penalty is the indicator

$$
P_\infty(\mathbf{x})=\begin{cases}0&\text{if }g_i(\mathbf{x})\le0\text{ for all }i,\\+\infty&\text{otherwise,}\end{cases}
$$

which makes $\min_\mathbf{x}f+P_\infty$ *exactly* the constrained problem. But it is discontinuous and useless for gradients. Practical penalties are compromises (Figure 6a shows each on the toy problem):
- **Quadratic**, $\mu\sum_i\max(0,g_i)^2$: smooth, but only approximately right. For the toy problem the minimizer of $x^2+\mu(1-x)^2$ is $x=\frac{\mu}{1+\mu}$: $0.5$ for $\mu=1$ and $0.91$ for $\mu=10$. It sits slightly **outside** the feasible set and reaches $x^*=1$ only as $\mu\to\infty$, where the problem becomes badly conditioned.
- **Exact $\ell_1$**, $\mu\sum_i\max(0,g_i)$: for $x<1$ the toy objective $x^2+\mu(1-x)$ has derivative $2x-\mu$, which is negative all the way to $x=1$ once $\mu\ge2$. So the minimizer is exactly $x^*=1$ for every $\mu\ge2$. The price is a kink at the solution. (The threshold $2$ is no accident: it is the multiplier $\lambda^*$ found below.)
- **Log barrier**, $-\frac1t\sum_i\log(-g_i)$: defined only strictly inside the feasible set, and blowing up at its edge. For the toy problem with $\frac1t=0.3$, setting the derivative of $x^2-0.3\log(x-1)$ to zero gives $2x(x-1)=0.3$, so $x=\frac{1+\sqrt{1.6}}{2}\approx1.13$, slightly **inside**. It approaches $x^*$ as $t\to\infty$. Following this path is the idea behind interior-point methods, the algorithm inside solvers such as cvxopt.[^nocedal2006]

![[Probability and Linear Algebra/images/multivariable-calculus-penalties.svg]]
*Figure 6. The toy problem $\min x^2$ subject to $x\ge1$ (feasible region shaded). (a) Penalized objectives and their minimizers (dots). The quadratic penalty's minimizer approaches $x^*=1$ from outside as $\mu$ grows ($0.5$, then $0.91$); the exact $\ell_1$ penalty hits it exactly; the log barrier approaches from inside ($1.13$). (b) The Lagrangian $x^2+\lambda(1-x)$ for several prices $\lambda$, with its unconstrained minimizer $x=\lambda/2$ marked. Only at $\lambda^*=2$ does that minimizer coincide with the constrained optimum.*

### The Lagrangian and duality

**Idea 2: a linear penalty, with an adversarial price.** Use the simplest penalty of all, a *linear* one, $\lambda_i\,g_i(\mathbf{x})$ with a price $\lambda_i\ge0$. Then let an adversary choose the prices to make the penalty as large as possible. For one constraint,

$$
\max_{\lambda\ge0}\ \lambda\,g(\mathbf{x})=\begin{cases}0&\text{if }g(\mathbf{x})\le0\quad(\text{best choice: }\lambda=0),\\+\infty&\text{if }g(\mathbf{x})>0\quad(\text{let }\lambda\to\infty).\end{cases}
$$

The inner maximization **rebuilds the ideal indicator penalty exactly**, out of smooth linear pieces. With the **Lagrangian**

$$
\mathcal{L}(\mathbf{x},\boldsymbol\lambda)=f(\mathbf{x})+\sum_{i=1}^m\lambda_i\,g_i(\mathbf{x}),
$$

this says

$$
\min_\mathbf{x}\ \max_{\boldsymbol\lambda\ge\mathbf{0}}\ \mathcal{L}(\mathbf{x},\boldsymbol\lambda)=\min_\mathbf{x}\ \big[f(\mathbf{x})+P_\infty(\mathbf{x})\big],
$$

which is the constrained problem.

Read $\lambda_i$ as the **price** of violating constraint $i$: the adversary raises the price of any violated constraint until we stop violating it. An equality constraint $h(\mathbf{x})=0$ works the same way with an *unrestricted* multiplier $\nu\in\mathbb{R}$, because $\max_\nu\nu\,h(\mathbf{x})$ is $0$ if $h=0$ and $+\infty$ otherwise. That is the Lagrangian of the previous section. (There we wrote $f-\lambda(g-c)$ for a maximization; with an unrestricted multiplier the sign convention is a matter of taste. For inequalities in the standard form above it is not: $\lambda_i\ge0$ and a plus sign.)

**Swapping min and max gives the dual.** Define the **dual function** $d(\boldsymbol\lambda)=\min_\mathbf{x}\mathcal{L}(\mathbf{x},\boldsymbol\lambda)$, the best we can do when the prices are fixed in advance. Because $\mathcal{L}$ is linear in $\boldsymbol\lambda$, $d$ is a minimum of linear functions, so it is always **concave**, even when the original problem is not convex. Two facts:
- **Weak duality.** For any $\boldsymbol\lambda\ge\mathbf{0}$ and any feasible $\mathbf{x}$, $d(\boldsymbol\lambda)\le\mathcal{L}(\mathbf{x},\boldsymbol\lambda)\le f(\mathbf{x})$, because the penalty terms $\lambda_ig_i$ are $\le0$ on the feasible set. So every dual value is a lower bound on the primal optimum $p^*$, and so is the best one, $d^*=\max_{\boldsymbol\lambda\ge\mathbf{0}}d(\boldsymbol\lambda)\le p^*$.
- **Strong duality.** If $f$ and all $g_i$ are convex and **Slater's condition** holds (some point is *strictly* feasible: $g_i(\mathbf{x})<0$ for all $i$), then $d^*=p^*$. There is then a **saddle point** $(\mathbf{x}^*,\boldsymbol\lambda^*)$: a minimum of $\mathcal{L}$ in $\mathbf{x}$ and a maximum in $\boldsymbol\lambda$, which solves both problems at once.[^boyd2004]

> [!example] Example: the toy problem's dual
> $\mathcal{L}(x,\lambda)=x^2+\lambda(1-x)$. For fixed $\lambda$, the minimum over $x$ is at $2x-\lambda=0$, so $x=\lambda/2$ (the dots in Figure 6b), and
>
> $$
> d(\lambda)=\frac{\lambda^2}{4}+\lambda\Big(1-\frac\lambda2\Big)=\lambda-\frac{\lambda^2}{4}.
> $$
>
> This concave parabola is maximized where $1-\lambda/2=0$, at $\lambda^*=2$, with $d^*=2-1=1$. That equals $f(x^*)=1$: strong duality, as Slater promises ($x=2$ is strictly feasible). At the right price, $x=\lambda^*/2=1$, the unconstrained minimizer of the Lagrangian **is** the constrained optimum.

The dual is often easier than the primal. The [[Support Vector Machines#The Lagrangian dual|SVM dual]] has simpler constraints than its primal, and it reveals that the data enter only through inner products, which is the door to kernels.

### The KKT conditions

At a saddle point $(\mathbf{x}^*,\boldsymbol\lambda^*)$, four conditions hold, named after Karush, Kuhn and Tucker:
1. **Stationarity.** $\mathbf{x}^*$ minimizes the *unconstrained* function $\mathcal{L}(\cdot,\boldsymbol\lambda^*)$, so the familiar test applies, but to $\mathcal{L}$ instead of $f$: $\nabla f(\mathbf{x}^*)+\sum_i\lambda_i^*\nabla g_i(\mathbf{x}^*)=\mathbf{0}$.
2. **Primal feasibility.** The inner max is finite, so $g_i(\mathbf{x}^*)\le0$ for all $i$.
3. **Dual feasibility.** $\lambda_i^*\ge0$. A negative price would *reward* violations.
4. **Complementary slackness.** $\lambda_i^*\,g_i(\mathbf{x}^*)=0$ for each $i$. The adversary maximizes $\lambda_ig_i$ with $g_i\le0$: if a constraint is slack ($g_i<0$), any positive price only lowers the penalty, so $\lambda_i^*=0$. A positive price is only possible on an **active** constraint ($g_i=0$).

Complementary slackness is the switch between the two situations described earlier: an inactive constraint gets $\lambda_i^*=0$ and drops out of stationarity, and an active one behaves like an equality constraint, with the extra requirement that its multiplier be non-negative. In the toy problem, $2x^*-\lambda^*=0$, $1-x^*\le0$, $\lambda^*=2\ge0$ and $\lambda^*(1-x^*)=0$ all hold at $x^*=1$.

Geometrically, stationarity says that $-\nabla f(\mathbf{x}^*)=\sum_i\lambda_i^*\nabla g_i(\mathbf{x}^*)$ is a non-negative combination of the gradients of the **active** constraints. Each $\nabla g_i$ points out of the feasible set (towards larger $g_i$), so every direction that would decrease $f$ points out of the feasible set: the constraints "push back" with forces $\lambda_i^*$. Compare (9), where the multiplier could have either sign because an equality constraint pushes from both sides.

> [!example] Example: KKT by hand, with the constraint active and inactive
> Minimize $(x-2)^2+(y-1)^2$, the squared distance to the point $(2,1)$, subject to $x+y\le1$.
> - **Try the constraint inactive** ($\lambda=0$): stationarity gives $(x,y)=(2,1)$. But $2+1=3>1$, infeasible. So the constraint must be active.
> - **Constraint active**: stationarity reads $2(x-2)+\lambda=0$ and $2(y-1)+\lambda=0$, so $x-2=y-1$, that is, $x=y+1$. With $x+y=1$ this gives $(x,y)=(1,0)$, and $\lambda=2(2-1)=2\ge0$. All four conditions hold, and the minimum value is $1+1=2$. Geometrically, $(1,0)$ is the projection of $(2,1)$ onto the line $x+y=1$.
>
> If the constraint were $x+y\le4$ instead, the unconstrained minimizer $(2,1)$ would be feasible, the constraint slack, and $\lambda=0$: complementary slackness at work.

**Why a zero gradient is no longer enough.** In unconstrained convex optimization, $\nabla f=\mathbf{0}$ alone certifies optimality. In constrained optimization, stationarity of $\mathcal{L}$ alone does not: $\nabla_\mathbf{x}\mathcal{L}=\mathbf{0}$ holds for *every* $\lambda$ at that $\lambda$'s minimizer (the dots in Figure 6b), and most of those points are infeasible or suboptimal. Only together with feasibility, non-negative prices and complementary slackness does it pin down the optimum. For convex problems satisfying Slater's condition, the four KKT conditions are **necessary and sufficient**, the constrained analogue of "$\nabla f=\mathbf{0}$ if and only if optimal". For non-convex problems they are only necessary, and only under a constraint qualification such as linearly independent active-constraint gradients.

[[Support Vector Machines#The Lagrangian dual|Support Vector Machines]] applies these four conditions to the maximum-margin problem: stationarity writes the weight vector as a combination of training points, dual feasibility makes the coefficients non-negative, and complementary slackness says only the points on the margin, the support vectors, can have non-zero coefficients. For a visual walkthrough of the KKT conditions and of how interior-point solvers use them, see Visually Explained's video on the topic.[^kktvideo]

## Multiple integrals and change of variables

### Iterated integrals and Fubini's theorem

A single integral adds up $f(x)\,dx$ over thin intervals. A **double integral** $\iint_Rf(x,y)\,dA$ adds up $f(x,y)$ times the area of tiny rectangles covering a region $R$; for $f\ge0$ it is the volume under the graph. The practical way to compute one is to slice: integrate over $y$ with $x$ held fixed (the area of one slice), then integrate the slices over $x$. **Fubini's theorem** says that the order does not matter, provided $\iint\lvert f\rvert\,dA<\infty$:

$$
\iint_Rf\,dA=\int\Big(\int f(x,y)\,dy\Big)dx=\int\Big(\int f(x,y)\,dx\Big)dy.
$$

> [!example] Example: a rectangle, both ways
> Integrate $f=xy^2$ over $0\le x\le1$, $0\le y\le2$.
> - $y$ first: $\int_0^2xy^2\,dy=x\cdot\frac83$, then $\int_0^1\frac83x\,dx=\frac43$.
> - $x$ first: $\int_0^1xy^2\,dx=\frac{y^2}{2}$, then $\int_0^2\frac{y^2}{2}\,dy=\frac{8}{6}=\frac43$.
>
> The same, as promised. Because $f$ factorizes as $x\cdot y^2$ over a rectangle, it is even simpler: $\int_0^1x\,dx\cdot\int_0^2y^2\,dy=\frac12\cdot\frac83$. This is the calculation that makes independent random variables easy ([[Joint Distributions#Independence|Joint Distributions]]).

> [!example] Example: a triangle, where the limits change
> Integrate $f=y$ over the triangle $0\le y\le x\le1$.
> - $y$ first: for fixed $x$, $y$ runs from $0$ to $x$: $\int_0^1\int_0^xy\,dy\,dx=\int_0^1\frac{x^2}{2}dx=\frac16$.
> - $x$ first: for fixed $y$, $x$ runs from $y$ to $1$: $\int_0^1\int_y^1y\,dx\,dy=\int_0^1y(1-y)\,dy=\frac12-\frac13=\frac16$.
>
> On non-rectangular regions the inner limits depend on the outer variable. Sketching the region first is the reliable way to get them right.

> [!warning] When Fubini fails
> The condition $\iint\lvert f\rvert<\infty$ matters. For $f(x,y)=\dfrac{x^2-y^2}{(x^2+y^2)^2}$ on the unit square, integrating over $y$ first and then $x$ gives $\frac\pi4$, while the other order gives $-\frac\pi4$. The function blows up at the corner $(0,0)$ in a way that is not absolutely integrable, and the two orders add up its positive and negative parts differently.

### Change of variables: the Jacobian determinant scales volume

In one variable, substituting $x=g(u)$ gives $\int f(x)\,dx=\int f(g(u))\,\lvert g'(u)\rvert\,du$: the factor $\lvert g'\rvert$ accounts for how $g$ stretches small intervals. In several variables, if $\mathbf{x}=\mathbf{T}(\mathbf{u})$ is a smooth one-to-one map from a region $D$ onto $\mathbf{T}(D)$,

$$
\int_{\mathbf{T}(D)}f(\mathbf{x})\,d\mathbf{x}=\int_Df\big(\mathbf{T}(\mathbf{u})\big)\,\big\lvert\det\mathbf{J}_\mathbf{T}(\mathbf{u})\big\rvert\,d\mathbf{u}.
\tag{10}
$$

The reason is the linear approximation. A tiny box of volume $d\mathbf{u}$ at $\mathbf{u}$ is mapped, approximately, by the linear map $\mathbf{J}_\mathbf{T}(\mathbf{u})$, and a linear map multiplies every volume by the same factor, the absolute value of its determinant. So the image of the tiny box has volume $\lvert\det\mathbf{J}_\mathbf{T}\rvert\,d\mathbf{u}$, and (10) just says that we must count each piece of $\mathbf{T}(D)$ with its true size.

> [!example] Example: a linear map
> $\mathbf{A}=\begin{bmatrix}2&1\\0&3\end{bmatrix}$ sends the unit square to the parallelogram spanned by its columns $(2,0)$ and $(1,3)$. Its area is base times height, $2\times3=6$, and $\det\mathbf{A}=2\cdot3-1\cdot0=6$ (Figure 7a). Every region, not just the square, has its area multiplied by 6. If $\mathbf{u}$ is uniform on the unit square (density 1), then $\mathbf{x}=\mathbf{A}\mathbf{u}$ is uniform on the parallelogram with density $\frac16$: densities shrink by the same factor that areas grow, so that the total probability stays 1.

> [!tip] Intuition
> 3Blue1Brown's *The determinant* shows the determinant as the factor by which a linear map scales area, and its sign as orientation.[^3b1b-det]

### Polar coordinates and the Gaussian integral

For polar coordinates we computed $\det\mathbf{J}_\mathbf{P}=r$ in [[#Worked example: polar coordinates]], so

$$
dx\,dy=r\,dr\,d\theta.
$$

Geometrically, the cell between radii $r$ and $r+dr$ and angles $\theta$ and $\theta+d\theta$ is nearly a rectangle with sides $dr$ and $r\,d\theta$ (Figure 7b). Cells far from the origin are bigger, and the factor $r$ accounts for that.

> [!example] Example: two integrals over a disk
> - The area of the disk of radius $R$: $\int_0^{2\pi}\int_0^Rr\,dr\,d\theta=2\pi\cdot\frac{R^2}{2}=\pi R^2$.
> - $\iint_{\text{disk}}(x^2+y^2)\,dA=\int_0^{2\pi}\int_0^Rr^2\cdot r\,dr\,d\theta=2\pi\cdot\frac{R^4}{4}=\frac{\pi R^4}{2}$. In Cartesian coordinates this would need $\int_{-R}^{R}\int_{-\sqrt{R^2-x^2}}^{\sqrt{R^2-x^2}}\cdots$, a much messier calculation. Choosing coordinates that match the symmetry of the problem is the whole art.

The most famous use of polar coordinates is the **Gaussian integral** $I=\int_{-\infty}^{\infty}e^{-x^2}\,dx$. The function $e^{-x^2}$ has no elementary antiderivative, so there is no direct route. The trick is to square $I$ and turn it into a double integral, using Fubini for the factorized integrand:

$$
\begin{aligned}
I^2&=\int_{-\infty}^{\infty}e^{-x^2}dx\int_{-\infty}^{\infty}e^{-y^2}dy\\
&=\iint_{\mathbb{R}^2}e^{-(x^2+y^2)}\,dx\,dy\\
&=\int_0^{2\pi}\int_0^\infty e^{-r^2}\,r\,dr\,d\theta=2\pi\Big[-\tfrac12e^{-r^2}\Big]_0^\infty=\pi.
\end{aligned}
$$

The factor $r$ from the Jacobian is exactly what makes the inner integral elementary: $r\,e^{-r^2}$ is the derivative of $-\frac12e^{-r^2}$. So

$$
\int_{-\infty}^{\infty}e^{-x^2}\,dx=\sqrt\pi,\qquad\int_{-\infty}^{\infty}e^{-x^2/2}\,dx=\sqrt{2\pi},
$$

the second by the one-dimensional substitution $x=\sqrt2\,u$. That $\sqrt{2\pi}$ is the normalizing constant of the standard Gaussian density $\frac{1}{\sqrt{2\pi}}e^{-x^2/2}$.

The same reasoning normalizes the multivariate Gaussian. Write $\mathbf{x}=\boldsymbol\mu+\mathbf{L}\boldsymbol\epsilon$ with $\mathbf{L}\mathbf{L}^\top=\boldsymbol\Sigma$. The integral over $\boldsymbol\epsilon$ factorizes into $D$ copies of $\sqrt{2\pi}$, and (10) contributes $\lvert\det\mathbf{L}\rvert=\sqrt{\det\boldsymbol\Sigma}$. That is where the $(2\pi)^{D/2}\lvert\boldsymbol\Sigma\rvert^{1/2}$ in the density of [[Joint Distributions#The multivariate Gaussian|the multivariate Gaussian]] comes from.

![[Probability and Linear Algebra/images/multivariable-calculus-area-scaling.svg]]
*Figure 7. Jacobian determinants scale area. (a) The linear map $\mathbf{A}=[[2,1],[0,3]]$ sends the unit square (blue, area 1) to a parallelogram (orange) of area $\det\mathbf{A}=6$. (b) A polar grid with equal steps in $r$ and $\theta$. The highlighted cell is nearly a rectangle with sides $\Delta r$ and $r\,\Delta\theta$, so its area is about $r\,\Delta r\,\Delta\theta$; cells further out are larger.*

### Densities under a change of variables

Formula (10) is also how densities transform. If $\mathbf{x}$ has density $p_X$ and $\mathbf{y}=\mathbf{T}(\mathbf{x})$ for an invertible smooth $\mathbf{T}$, then

$$
p_Y(\mathbf{y})=p_X\big(\mathbf{T}^{-1}(\mathbf{y})\big)\,\big\lvert\det\mathbf{J}_{\mathbf{T}^{-1}}(\mathbf{y})\big\rvert.
$$

Where $\mathbf{T}$ stretches space, the same probability is spread over more volume and the density drops; where it squeezes, the density rises. The parallelogram example above is the simplest case. In one dimension this is the change-of-variables formula of [[Random Variables#Functions of a random variable|Random Variables]], and in many dimensions it is the foundation of [[Normalizing Flows|normalizing flows]], which build flexible densities by composing invertible maps and keeping track of exactly this determinant.

## Key takeaways

- A derivative is the best linear approximation. In many variables, the gradient (scalar output) or the Jacobian (vector output) plays that role: $f(\mathbf{a}+\mathbf{h})\approx f(\mathbf{a})+\nabla f^\top\mathbf{h}$. Partial derivatives existing is not enough; continuous partials are.
- The directional derivative is $\nabla f^\top\mathbf{u}$. Hence the gradient points in the direction of steepest ascent, with slope $\lVert\nabla f\rVert$, and it is perpendicular to the level sets.
- The chain rule multiplies Jacobians. Backpropagation evaluates that product from the loss backwards, so that every intermediate result is a cheap row vector.
- The Hessian is symmetric (Schwarz) and gives the quadratic model $f(\mathbf{a}+\mathbf{h})\approx f(\mathbf{a})+\nabla f^\top\mathbf{h}+\frac12\mathbf{h}^\top\mathbf{H}\mathbf{h}$. At a critical point, its eigenvalues decide: all positive is a minimum, all negative a maximum, mixed signs a saddle, and zeros make the test inconclusive. For $2\times2$, check the determinant and the trace.
- Matrix calculus: expand $f(\mathbf{x}+\mathbf{h})$ and read off the linear term. $\nabla\mathbf{a}^\top\mathbf{x}=\mathbf{a}$, $\nabla\mathbf{x}^\top\mathbf{A}\mathbf{x}=(\mathbf{A}+\mathbf{A}^\top)\mathbf{x}$, $\nabla\lVert\mathbf{A}\mathbf{x}-\mathbf{b}\rVert^2=2\mathbf{A}^\top(\mathbf{A}\mathbf{x}-\mathbf{b})$ (the normal equations), $\nabla\operatorname{lse}=\operatorname{softmax}$, $\nabla\log\det\mathbf{X}=\mathbf{X}^{-\top}$. Always check gradients numerically.
- Convex functions lie above their tangents and have positive semidefinite Hessians. For them, a zero gradient means a global minimum, and every local minimum is global. Strong convexity adds a unique minimizer and quadratic growth.
- With an equality constraint, $\nabla f=\lambda\nabla g$ at the optimum, and $\lambda$ is the price of the constraint. With inequalities, the Lagrangian's inner maximum rebuilds the ideal penalty; the dual gives lower bounds, which are tight for convex problems under Slater's condition; and the four KKT conditions replace "gradient equals zero".
- In integrals, change of variables multiplies by $\lvert\det\mathbf{J}\rvert$, the local volume scaling. Polar coordinates give $dx\,dy=r\,dr\,d\theta$ and $\int e^{-x^2}dx=\sqrt\pi$.

## Where this shows up next

- [[Orthogonality and Least Squares]] derives the normal equations again from geometry, and [[Eigenvalues and Eigenvectors]] develops the positive definite matrices behind the Hessian test.
- [[First-Order Optimization]] turns the gradient and the quadratic model into gradient descent, and uses the Hessian's eigenvalues and strong convexity to explain how fast it converges.
- [[Linear Models]] minimizes $\lVert\mathbf{X}\mathbf{w}-\mathbf{y}\rVert^2$ and the softmax cross-entropy with the gradients derived here; [[Neural Networks]] computes gradients of deep compositions by backpropagation.
- [[Support Vector Machines]] is an application of the Lagrangian, duality and the KKT conditions; [[Matrix Factorization and Dimensionality Reduction]] finds principal components with a Lagrange multiplier.
- [[Random Variables]] and [[Joint Distributions]] use change of variables for densities and the Gaussian integral for normalization; [[Energy-Based Models]] starts from the maximum-entropy Boltzmann distribution.

<!-- References: footnote definitions below render as the numbered "References" list at the bottom of the page -->

[^mml2020]: Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). *Mathematics for Machine Learning*, Chapters 5 (vector calculus) and 7 (continuous optimization). Cambridge University Press. [book website](https://mml-book.github.io/)
[^3b1b-chain]: 3Blue1Brown. *Visualizing the chain rule and product rule | Chapter 4, Essence of calculus*. YouTube. [video](https://www.youtube.com/watch?v=YG15m2VwSjA)
[^3b1b-taylor]: 3Blue1Brown. *Taylor series | Chapter 11, Essence of calculus*. YouTube. [video](https://www.youtube.com/watch?v=3d6DsjIBzJ4)
[^parr2018]: Parr, T., & Howard, J. (2018). *The Matrix Calculus You Need For Deep Learning*. arXiv:1802.01528. [arXiv](https://arxiv.org/abs/1802.01528)
[^cookbook]: Petersen, K. B., & Pedersen, M. S. (2012). *The Matrix Cookbook*. Technical University of Denmark. [PDF](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
[^boyd2004]: Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, Chapters 2–3 (convex sets and functions) and 5 (duality). Cambridge University Press. [book website](https://web.stanford.edu/~boyd/cvxbook/)
[^jaynes1957]: Jaynes, E. T. (1957). *Information Theory and Statistical Mechanics*. Physical Review, 106(4), 620–630. [doi:10.1103/PhysRev.106.620](https://doi.org/10.1103/PhysRev.106.620)
[^nocedal2006]: Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.), Chapters 12 (constrained optimization theory), 17 (penalty methods) and 19 (interior-point methods). Springer. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
[^kktvideo]: Visually Explained. *The Karush–Kuhn–Tucker (KKT) Conditions and the Interior Point Method for Convex Optimization*. YouTube. [video](https://www.youtube.com/watch?v=uh1Dk68cfWs)
[^3b1b-det]: 3Blue1Brown. *The determinant | Chapter 6, Essence of linear algebra*. YouTube. [video](https://www.youtube.com/watch?v=Ip3X9LOh2dk)
