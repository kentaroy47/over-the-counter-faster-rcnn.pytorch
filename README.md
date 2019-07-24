# what's this?
simple over the counter faster rcnn!
works with pytorch 1.0

# Install
pip install -r requirements.txt

```
cd lib
python setup.py build develop
```

# Inference

This will do a res18 inference of your images.

The model is trained by MS-COCO.
```
python inference.py --image_dir path-to-your-image-dir
```
