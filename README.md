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
git clone https://github.com/aherbert/mni.git
# Change into the project directory
cd mni
# Create and activate virtual environment
uv sync
source .venv/bin/activate
```

## Updating

```bash
# Pull the latest changes
git pull
# Resync the virtual environment
uv sync
```

## Usage

### Cellpose models

Segmentation uses `cellpose` which requires that the named model be installed in the
`cellpose` models directory. For custom models this can be achieved using:

        cellpose --add_model [model path]

The default cellpose 4 model is `cpsam` (no  install required). This works well for
typical nuclei images.

### Analysis of images

Analysis of micro-nuclei images requires a CYX input image. This can be in TIFF or CZI
(Carl Zeiss Image) format. The script `mni-analysis.py` is used to run the analysis:

```bash
# Activate the environment (if not active)
source .venv/bin/activate

# Analyse
./mni-analysis.py /path/to/image.[tiff|czi]
```

This script will perform the following steps:

1. Segment the selected nuclei channel.
1. Identify spots in the two selected spot channels.
1. Assign nuclei objects to classes (nucleus; micro-nucleus; bleb).
Create analysis objects and for each group compute spot counts, and between channel
spot overlap and nearest neighbours.

Results are saved to files with the same prefix as the input image:

- `.objects.tiff`: Nuclei label mask.
- `.spot1.tiff`: Spot channel 1 label mask.
- `.spot2.tiff`: Spot channel 2 label mask.
- `.spots.csv`: Spot details table.
- `.summary.csv`: Nuclei summary table.
- `.settings.json`: JSON file with the runtime settings.

The results can be visualised in `Napari` using the `--view` option. This will load
the image as channels and the 3 label layers. The results tables are associated with
the appropriate label layers. The editing tools within `Napari` can be used to update
the label masks, e.g. add or remove spots; update the nuclei objects. A widget
within `Napari` allows the results tables to be regenerated from modified labels. This
will save the current label layers to file allowing the analysis to be continued in
a subsequent session by reloading the results:

```bash
# [Re]Analyse and view
./mni-analysis.py /path/to/image.[tiff|czi] --view
```

Analysis can be repeated which will reload existing results or restart the analysis at the
given stage, e.g. 1; 2; or 3.

### Multiple images

Multiple images can be passed as arguments to the analysis script. It is also possible to
pass in directories. In this case the script will run on any file with the CZI extension
and any TIFF file containing a CYX image. This allows the script to run on a directory
containing existing result masks as these YX images will be ignored. The CZI images are
expected to be CYX or CYX0 (where the last dimension is the sample).

```bash
# Analyse an image and two image directories
./mni-analysis.py /path/to/image.[tiff|czi] /path/to/images/ /path/to/more/images/ --view
```

### Reporting

The results CSV files can be collated and used to generate reports across the analysis.
This can be done by passing individual CSV files or results directories. Only files
ending `.spots.csv` or `.summary.csv` are loaded. Reports are printed to the console
and saved to the specified output directory. Reports can be selected or by default
all reports are generated. Use the help option (`-h`) to view parameters that change the
report queries.

```bash
# Generate reports on the MNi analysis
./mni-reports.py /path/to/images/ /path/to/more/images/
```

## Development

This project uses [pre-commit](https://pre-commit.com/) to create actions to validate
changes to the source code for each `git commit`.
Install the hooks for your development repository clone using:

    pre-commit install

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
