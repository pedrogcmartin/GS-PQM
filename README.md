<div align="center">
   <h1>GS-PQM: A Parameter-Domain Quality Metric for Compressed Gaussian Splatting</h1>
   <br />

   Pedro Martin, António Rodrigues, João Ascenso, Maria Paula Queluz

   Instituto de Telecomunicações, Instituto Superior Técnico, University of Lisbon
   <br />

  <p align="center">
   <img src="imgs/pipeline.jpeg">
    <br />
  </p>
</div>
GS-PQM is a novel full-reference quality metric for post-training GS compression that operates directly in the GS parameter domain. GS-PQM estimates perceptual quality from a set of parameter-domain distortion errors (GS-Dist) using a Support Vector Regression model.

## Cloning the Repository

Clone the GS-PQM repository and enter the project directory:

```bash
git clone https://github.com/pedrogcmartin/GS-PQM.git
cd GS-PQM
```

## Installation

We recommend using a dedicated Conda environment to install the required dependencies:

```bash
conda create -n gspqm python=3.9
conda activate gspqm
pip install -r requirements.txt
```

The pretrained GS-PQM model is included in the models/ directory, so no training is required to use the metric.

## Running

GS-PQM requires an uncompressed reference GS model and the corresponding compressed/distorted GS model. Both models must be provided as .ply files.

To compute the GS-PQM quality score, run:

```bash
python gspqm.py --ref path/to/reference/point_cloud.ply --dist path/to/compressed/point_cloud.ply
```

Use the `--verbose` option to additionally display the GS-Dist errors and the adaptive-neighborhood statistics.

## Examples

The `examples/` directory contains a reference GS model and its corresponding compressed version:

```text
examples/
├── ref/
│   └── point_cloud.ply
└── dist/
    └── point_cloud.ply
```

To run GS-PQM on the provided example, use:

```bash
python gspqm.py --ref examples/ref/point_cloud.ply --dist examples/dist/point_cloud.ply
```

The expected output is:

```bash
GS-PQM predicted DMOS: 4.888012
```