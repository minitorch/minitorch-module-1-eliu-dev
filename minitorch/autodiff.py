from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # Create lists from the input tuple for easy modification
    vals_plus = list(vals)
    vals_minus = list(vals)
    
    # Modify the arg-th element for both forward and backward difference
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon
    
    # Calculate f(x + epsilon) and f(x - epsilon)
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)
    
    # Return the central difference
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = set()
    result = []
    def visit(v: Variable):
        if v.unique_id not in visited:
            visited.add(v.unique_id)
        else:
            return
        for parent in v.parents:
            if not parent.is_constant():
                visit(parent)
        result.append(v)
    
    visit(variable)
    result.reverse()
    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave vars.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    sorted_var = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}
    
    for var in sorted_var:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
            print(f'Unique ID: {var.unique_id}, Derivative: {var.derivative}, Derivatives: {derivatives[var.unique_id]}', flush=True)
        else:
            for parent, grad in var.chain_rule(derivatives[var.unique_id]):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = grad
                else:
                    derivatives[parent.unique_id] += grad
    print(derivatives, flush=True)

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
