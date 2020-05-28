#Image Mixer using Neural Style Transfer - Tensorflow

This is a simple Flask app takes two images as input, a base image and a style image as input. In the backend, a model is trained to generate a series of mixed images, which are later combined into a single gif image.

## Installation

Clone the repository and create a virtual environment if you need. Install dependencies using,

```bash
pip install -r requirements.txt
```

## Usage

To enable debug mode on windows use,

```bash
set FLASK_ENV=development
set debug=True
```
on Ubuntu/Mac OS use,

```bash
export FLASK_ENV=development
export debug=True
```

Then, run it with,

```bash
flask run
```

After selecting the two input images, it takes about a while - around 7 minute on my device atleast, to train and get the output. You might as well change the value of _epochs_ and _steps_per_epoch_

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
