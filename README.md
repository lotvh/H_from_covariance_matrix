# methods to approximate parent hamiltonian from covariance matrix
## using gradient descent

This method is implemented in `gradient_descent_find_hamiltonian.py`

### Objective

Using the $L_2$ loss function which provides a measure of how well a candidate
Hamiltonian $H^{(c)}$ approximates a target covariance matrix $\Gamma^{(t)}$,
$$
L\left(H^{(c)}, \Gamma^{(t)}\right) = \sqrt{
    \sum_{i,j}^n\left(
        \Gamma_{H^{(c)}} - \Gamma^{(t)}
    \right)^2
} \ ,
$$
as well as some parametrization of the Hamiltonian $H(\left\{P_i\right\})$ with
parameters $P_i$, the problem of finding the parent hamiltonian for a given
target covariance matrix $\Gamma^{(t)}$ can be formulated as

$$ H \approx \mathop{\mathrm{arg\,min}}_{\{P_i\}}
L\left(H\left(\left\{P_i\right\}\right), \Gamma^{(t)}\right) \ . $$

Gradient descent is a well-known method to find
a solution for this argument. It is an iterative process, in which normally
speaking a candidate
Hamiltonian $H^{(c)}$ is improved in each iteration step given by

$$ H^{(c)}_{n+1} = H^{(c)}_{n} - \eta \nabla_{\left\{P_i\right\}}
L\left(H^{(c)}_{n}(\left\{P_i\right\}), \Gamma^{(t)}\right) \ . $$

Here, $\nabla_{\{P_i\}}$ denotes the gradient to all parameters of the
Hamiltonian, and $\eta$ is a learning rate determining the step size.

While in most machine learning contexts gradients can easily be found
analytically, this is not the case for the process of finding the covariance
matrix $\Gamma$ outlined in section \ref{sec:freefermionfields}, since it
involves finding the symplectic eigenvalues of the Hamiltonian.

Instead, a numeric approximation of the gradient is performed by adding
a peturbation $\epsilon$ to each parameter, i.e.

$$ H^{(p)}_j = H\left(\left\{P_j+\epsilon, P_i|i \neq j\right\}\right) \ . $$

Defining the basis of perturbations of the covariance matrix
$\{\Delta\Gamma^{(p)}_i\}$ with $$ \Delta\Gamma^{(p)}_i = \Gamma_{H^{(p)}_i}
- \Gamma^{(c)} \ , $$
one can represent the difference between the covariance matrix
found from the candidate Hamiltonian and the target covariance matrix
$\Delta\Gamma^{(c)} = \Gamma_{H^{(c)}} - \Gamma^{(t)}$ in this basis.
The gradient of the loss function is then approximately proportional to the
coefficients of this representation.

Since the exact proportionality is not known, normalization of the candidate
hamiltonian has to be performed on each iteration to prevent unbounded growth of
parameters. As a normalization coefficient, using the difference between the
minimal and maximal matrix elements of the candidate hamiltonian has shown good
results.

Validation of this method is performed by finding the covariance matrix of
a known parent hamiltonian, then attempting to approximate the parent
hamiltonian. The validation results in Figure \ref{fig:gradient_validation}
indicate that approximate parent Hamiltonians can successfully be recovered from
a covariance matrix using the gradient descent method presented above.

### Libraries Used
- numpy
- scipy
- matplotlib

## using convex optimization

This method is implemented in `cvxpy_find_hamiltonian.py`.

### Objective

A second method to approximate the parent hamiltonian for a given covariance
matrix, is convex optimization.

### Libraries used
- cvxpy
- numpy
- scipy
- math
- sys
- matplotlib
