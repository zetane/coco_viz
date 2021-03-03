# coco_viz

## Requirements

* `git`
* `git-lfs` [extension to Git](https://git-lfs.github.com/).
* Zetane [Viewer](https://zetane.com/gallery)
* Python 3
* Install the module requirements from `requirements.txt` (PyPi listing). This includes the Zetane Python module.


## Setup

The repository refers to Coco and datasets. They need a one-time initialization:

```
git submodule init
git lfs pull
pip install -r requirements.txt
```

The Git submodule integrates the Coco API, and LFS gets the datasets.


## Run

First, please start the Zetane viewer. This should open the viewer on a blank universe. Then:

```
python test_torch.py
```

The script will use the datasets to train a model and generate visualizations in the Zetane Viewer.


## Legacy Documentation

Run the sample

```git submodule init # fetches cocoapi
sh extract.sh      # extracts small subset of validation data
cd cocoapi/PythonAPI
python setup.py build_ext install # installs the cocoapi package in your venv or conda env
```

Run Zetane Engine via command line:
```
open /Applications/Zetane.app --args --server
```

Run sample script
```
python torch_test.py
```
