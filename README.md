# coco_viz
To run the sample:
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
