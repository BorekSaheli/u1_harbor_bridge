"""
Simple API for multi-stakeholder optimization problems.
Clean, intuitive interface with string-based expressions.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.interpolate import pchip_interpolate
from genetic_algorithm_pfm import GeneticAlgorithm


class Result:
    """Container for optimization results."""

    def __init__(self, variables, score, variable_names, stakeholders_data, problem):
        self.variables = dict(zip(variable_names, variables))
        self.variables_array = variables
        self.score = score
        self.stakeholders = stakeholders_data
        self.problem = problem

    def plot(self, paradigm=''):
        """Plot the preference curves with optimal solutions."""
        n_stakeholders = len(self.stakeholders)

        # Create 2-row layout: 3 plots in first row, remaining in second row
        if n_stakeholders <= 3:
            # If 3 or fewer, use single row
            fig, axes = plt.subplots(1, n_stakeholders, figsize=(5*n_stakeholders, 5))
            if n_stakeholders == 1:
                axes = [axes]
            axes_flat = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            # Use 2 rows: 3 in first row, rest in second
            n_cols = 3
            fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
            axes_flat = axes.flatten()

            # Hide unused subplots in second row if n_stakeholders < 6
            for i in range(n_stakeholders, len(axes_flat)):
                axes_flat[i].axis('off')

        # Add main title
        if paradigm:
            fig.suptitle(f'Multi-Stakeholder Optimization Results - {paradigm.upper()} Paradigm',
                        fontsize=16, fontweight='bold', y=0.995)

        for i, (name, data) in enumerate(self.stakeholders.items()):
            ax = axes_flat[i]

            # Get preference curve points and x-label
            stakeholder_info = self.problem.stakeholders[i]
            x_points, y_points = stakeholder_info['preference_points']
            x_label = stakeholder_info.get('x_label', 'Objective value')

            # Create continuous curve
            x_range = np.linspace(min(x_points), max(x_points), 100)
            y_range = pchip_interpolate(x_points, y_points, x_range)

            # Plot curve
            ax.plot(x_range, y_range, label='Preference curve', color='black')

            # Plot result point
            ax.scatter(data['objective_value'], data['preference_score'],
                      label='Optimal solution', color='orange', marker='o', s=100, zorder=5)

            ax.set_xlim((min(x_points), max(x_points)))
            ax.set_ylim((0, 102))
            ax.set_title(name)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Preference score')
            ax.grid(linestyle='--')
            ax.legend()

        fig.tight_layout()
        plt.show()

    def print_summary(self):
        """Public method to print optimization summary."""
        self._print_summary()

    def _print_summary(self):
        """Print a clean summary of the optimization problem and results."""
        print("\n" + "="*80)
        print("OPTIMIZATION SUMMARY")
        print("="*80)

        # Variables
        print("\nDESIGN VARIABLES:")
        for var_name, var_value in self.variables.items():
            # Add description if available
            description = self.problem.variable_descriptions.get(var_name, '')
            if description:
                print(f"   {var_name} = {var_value:.4f}  ({description})")
            else:
                print(f"   {var_name} = {var_value:.4f}")

        print(f"\nAGGREGATE SCORE: {self.score:.4f}")

        # Stakeholders and their objectives
        print("\nSTAKEHOLDERS & OBJECTIVES:")
        print("-" * 80)
        for i, (name, data) in enumerate(self.stakeholders.items()):
            stakeholder_info = self.problem.stakeholders[i]
            print(f"\n   {name}:")
            print(f"      Objective function: {stakeholder_info['objective']}")
            print(f"      Weight: {stakeholder_info['weight']:.2f}")
            print(f"      Objective value: {data['objective_value']:.4f}")
            print(f"      Preference score: {data['preference_score']:.2f}/100")

        # Constraints
        if self.problem.constraints:
            print("\nCONSTRAINTS:")
            print("-" * 80)
            for i, constraint in enumerate(self.problem.constraints, 1):
                print(f"   {i}. {constraint}")

        print("\n" + "="*80 + "\n")


class Problem:
    """Main problem class for multi-stakeholder optimization."""

    def __init__(self, variables, bounds, variable_descriptions=None):
        """
        Initialize the problem.

        :param variables: List of variable names (e.g., ['x1', 'x2', 'x3', 'x4'])
        :param bounds: Either a single [min, max] pair or list of pairs for each variable
        :param variable_descriptions: Optional dict mapping variable names to descriptions
                                      (e.g., {'x1': 'Vertical clearance [meters]'})
        """
        self.variable_names = variables
        self.n_variables = len(variables)
        self.variable_descriptions = variable_descriptions or {}

        # Handle bounds
        if len(bounds) == 2 and not isinstance(bounds[0], list):
            # Single bound for all variables
            self.bounds = [bounds] * self.n_variables
        else:
            self.bounds = bounds

        self.stakeholders = []
        self.constraints = []

    def add_stakeholder(self, name, weight, objective, preference_points, x_label=None):
        """
        Add a stakeholder to the problem.

        :param name: Stakeholder name
        :param weight: Weight in aggregation (0-1)
        :param objective: String expression (e.g., "11.25*x1 + 13.75*x2")
        :param preference_points: Tuple of ([x_points], [y_points]) for pchip interpolation
        :param x_label: Optional label for x-axis in plots (e.g., "Cost [billion dollars]")
        """
        self.stakeholders.append({
            'name': name,
            'weight': weight,
            'objective': objective,
            'preference_points': preference_points,
            'x_label': x_label if x_label else 'Objective value'
        })

    def add_constraint(self, expression):
        """
        Add a constraint to the problem.

        :param expression: String with constraint (e.g., "x1 + x2 >= 200" or "x1 <= 100")
        """
        self.constraints.append(expression)

    def _parse_expression(self, expr_str):
        """
        Parse string expression into a function.

        :param expr_str: String expression with variable names
        :return: Function that takes array and returns result
        """
        # Create a function that evaluates the expression
        def expr_func(variables):
            # Create local namespace with variables
            local_vars = {}
            for i, var_name in enumerate(self.variable_names):
                local_vars[var_name] = variables[:, i]

            # Evaluate expression
            result = eval(expr_str, {"__builtins__": {}}, local_vars)
            return result

        return expr_func

    def _parse_constraint(self, constraint_str):
        """
        Parse constraint string into GA-compatible function.

        :param constraint_str: String like "x1 + x2 >= 200" or "x1 <= 100"
        :return: Function that returns value that should be < 0
        """
        # Split by comparison operator
        if '>=' in constraint_str:
            parts = constraint_str.split('>=')
            left = parts[0].strip()
            right = parts[1].strip()
            # For >=: convert to -(left) + right < 0
            expr_str = f"-({left}) + ({right})"
        elif '<=' in constraint_str:
            parts = constraint_str.split('<=')
            left = parts[0].strip()
            right = parts[1].strip()
            # For <=: convert to (left) - (right) < 0
            expr_str = f"({left}) - ({right})"
        elif '>' in constraint_str:
            parts = constraint_str.split('>')
            left = parts[0].strip()
            right = parts[1].strip()
            expr_str = f"-({left}) + ({right})"
        elif '<' in constraint_str:
            parts = constraint_str.split('<')
            left = parts[0].strip()
            right = parts[1].strip()
            expr_str = f"({left}) - ({right})"
        else:
            raise ValueError(f"Constraint must contain >=, <=, >, or <: {constraint_str}")

        return self._parse_expression(expr_str)

    def _objective_function(self, variables):
        """
        Internal objective function for the GA.

        :param variables: Array with design variable values per population member
        :return: Tuple of (weights, preference_scores)
        """
        weights = []
        preferences = []

        for stakeholder in self.stakeholders:
            weights.append(stakeholder['weight'])

            # Calculate objective value
            obj_func = self._parse_expression(stakeholder['objective'])
            obj_value = obj_func(variables)

            # Calculate preference score
            x_points, y_points = stakeholder['preference_points']
            pref_score = pchip_interpolate(x_points, y_points, obj_value)

            preferences.append(pref_score)

        return weights, preferences

    def solve(self, paradigm='minmax', n_iter=400, n_pop=500, n_bits=8,
              r_cross=0.8, max_stall=8, var_type='real'):
        """
        Solve the optimization problem.

        :param paradigm: Aggregation paradigm ('minmax' or 'tetra')
        :param n_iter: Number of iterations
        :param n_pop: Population size
        :param n_bits: Number of bits for encoding
        :param r_cross: Crossover rate
        :param max_stall: Maximum stall iterations
        :param var_type: Variable type ('real' or 'int')
        :return: Result object
        """
        # Parse constraints
        constraint_funcs = []
        for const_str in self.constraints:
            const_func = self._parse_constraint(const_str)
            constraint_funcs.append(['ineq', const_func])

        # GA options
        options = {
            'n_bits': n_bits,
            'n_iter': n_iter,
            'n_pop': n_pop,
            'r_cross': r_cross,
            'max_stall': max_stall,
            'aggregation': paradigm,
            'var_type': var_type
        }

        # Run GA
        ga = GeneticAlgorithm(
            objective=self._objective_function,
            constraints=constraint_funcs,
            bounds=self.bounds,
            options=options
        )

        score, design_variables, _ = ga.run()

        # Convert to numpy array if needed
        design_variables = np.array(design_variables)

        # Evaluate stakeholders
        stakeholders_data = {}
        for stakeholder in self.stakeholders:
            obj_func = self._parse_expression(stakeholder['objective'])

            # Create single-row array for evaluation
            single_var = design_variables.reshape(1, -1)
            obj_value = float(obj_func(single_var))

            x_points, y_points = stakeholder['preference_points']
            pref_score = float(pchip_interpolate(x_points, y_points, obj_value))

            stakeholders_data[stakeholder['name']] = {
                'objective_value': obj_value,
                'preference_score': pref_score
            }

        return Result(design_variables, score, self.variable_names,
                     stakeholders_data, self)


if __name__ == '__main__':
    # Set seed for reproducibility (same as chapter_11_1.ipynb)
    np.random.seed(42)

    # 1. Create problem
    problem = Problem(
        variables=['x1', 'x2', 'x3', 'x4'],
        bounds=[0, 260]
    )

    # 2. Add stakeholder 1: Project Developer
    problem.add_stakeholder(
        name="Project developer",
        weight=0.50,
        objective="11.25*x1 + 13.75*x2 + 15*x3 + 11.25*x4",
        preference_points=([3000, 3500, 4000], [0, 20, 100])
    )

    # 3. Add stakeholder 2: Municipality
    problem.add_stakeholder(
        name="Municipality",
        weight=0.50,
        objective="x1 + x4",
        preference_points=([100, 125, 160], [0, 50, 100])
    )

    # 4. Add constraints (natural syntax)
    problem.add_constraint("x1 + x2 + x3 + x4 >= 200")  # min 200 houses
    problem.add_constraint("x1 + x2 + x3 + x4 <= 260")  # max 260 houses
    problem.add_constraint("11.25*x1 + 13.75*x2 + 15*x3 + 11.25*x4 >= 3000")  # min profit
    problem.add_constraint("11.25*x1 + 13.75*x2 + 15*x3 + 11.25*x4 <= 4000")  # max profit
    problem.add_constraint("x1 + x4 >= 100")  # min affordable
    problem.add_constraint("x1 + x4 <= 150")  # max affordable

    # 5. Solve with both paradigms
    paradigms = ['minmax', 'tetra']

    for paradigm in paradigms:
        print(f'\n{"="*60}')
        print(f'Run GA with {paradigm}')
        print(f'{"="*60}')

        result = problem.solve(paradigm=paradigm)

        # 6. Print results
        print(f'\nOptimal result:')
        for var_name, var_value in result.variables.items():
            print(f'  {var_name} = {var_value:.2f} houses')

        print(f'\nScore: {result.score:.4f}')

        print(f'\nStakeholder results:')
        for name, data in result.stakeholders.items():
            print(f'  {name}:')
            print(f'    Objective value: {data["objective_value"]:.2f}')
            print(f'    Preference score: {data["preference_score"]:.2f}')

    # 7. Plot results for minmax
    print(f'\n{"="*60}')
    print('Plotting results for minmax paradigm...')
    print(f'{"="*60}\n')

    np.random.seed(42)
    result = problem.solve(paradigm='minmax')
    result.plot()
