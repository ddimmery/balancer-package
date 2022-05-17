## Contributing to this project

## Code of Conduct

Before starting out, please take a look at our [Code of Conduct](code_of_conduct.md). Participation means that you agree to engage constructively with the community as per the Code.

## Development

The first step is to clone this repository and install all dependencies. You can do this with [poetry](https://python-poetry.org/):

```
poetry install
```

or just with `pip`:
```
pip install .
```

Using poetry is particularly nice because it will keep all dependencies in a virtual environment without confusing your local setup.

A few tools are provided to make things easier. A basic `Makefile` provides the necessary commands to build the entire package and documentation. Running `make` will build everything necessary for local testing.

## Getting Started
Contributions are made to this repo via Issues and Pull Requests (PRs), primarily the former.

### Issues

Please try to provide a [minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example). If that isn't possible, explain as clearly and simply why that is, along with all of the relevant debugging steps you've already taken.

### Pull Requests (PRs)

Since this is a stripped down implementation, it seems unlikely we will accept substantial PRs or feature requests. If you believe there is fundamental functionality that is missing, feel free to open an Issue and we can discuss.