# Scheffé-Wald Test for Multiple Comparisons
This project implements a Scheffé-Wald test to perform multiple comparisons in statistical analysis.The test is designed to identify significant differences between group means while controlling the familywise error rate. The implementation is organized as a Python project using Poetry for dependency management and packaging.

# Project Structure

```js
tesis
├── MultipleTesting/                     # Main implementation 
│   ├── multiple_comparisons.py # Statistcal tests fucntions
│   └── visualization.py       # visualization functions
├── tests/                     # Test suite for the project
├── penalty_analysis
│    ├── data # data for practical example
│    └── implementation.ipynb
├── poetry.lock                
├── pyproject.toml                  
└── README.md            
```

# Features
Scheffé-Wald Test Implementation: Includes functions to perform the Scheffé-Wald test on datasets.
* Visualization: Tools for plotting test results and visualizing group comparisons.
* Test Suite: Comprehensive tests to validate functionality and ensure accuracy.
* Extensibility: Can be extended to support other multiple comparison tests.

# Requirements
* ``Python: >=3.8``
* **Dependencies:** Managed via Poetry. Install all dependencies with:
```bash
poetry install
``` 


# Usage
1. Install dependencies:

```bash
poetry install
```

2. Run the Scheffé-Wald test:
```bash
poetry run python tesis/run_scheffe_test.py --data input.csv --alpha 0.05
```

3. Visualize results:

Use the ``visualization.py`` script to generate plots of the test outcomes.
Example:
```bash
poetry run python visualization/plot_results.py --data results.csv
```

# Configuration
The project uses a ``pyproject.toml`` file for configuration. Adjust the following sections as needed:

* **Dependencies**: Add additional dependencies in the ``[tool.poetry.dependencies]`` section.
* **Scripts**: Update the ``[tool.poetry.scripts]`` section to add custom CLI commands.

# Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork this repository.
2. Create a feature branch:
```bash
git checkout -b feature-name
```
3. Commit your changes and push to the branch:
```bash
git commit -m "Add new feature"
git push origin feature-name
```
4. Open a pull request.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments
This project is inspired by statistical techniques for robust multiple comparisons, particularly the Scheffé method.