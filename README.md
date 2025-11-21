# MNi

Micro-nucleus analysis.

## Installation

```bash
# Install uv
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/aherbert/plate-stitch.git
# Change into the project directory
cd plate-stitch
# Create and activate virtual environment
uv sync
source .venv/bin/activate
```

## Usage

Segmentation uses `cellpose` which requires that the named model be installed in the
`cellpose` models directory. This can be achieved using:

        cellpose --add_model [model path]

## Development

This project uses [pre-commit](https://pre-commit.com/) to create actions to validate
changes to the source code for each `git commit`.
Install the hooks for your development repository clone using:

    pre-commit install

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
