# delphi
Python implementation of [DELPHI](https://www.frontiersin.org/articles/10.3389/fmars.2015.00020/full).

## Installation
```bash
pip install delphi-laser-detection
```

## Usage

`delphi` provides a command line interface for annotation, training, and detection.

### 1. Annotate

```bash
delphi annotate image_dir/ annotation_dir/
```

Click on the image to add a point. Press any key to save the annotations and move to the next image. Previous annotations will be loaded if they exist.

### 2. Train

```bash
delphi train annotation_dir/ annotation_dir/ model_dir/
```

### 3. Detect

```bash
delphi detect image_dir/ annotation_dir/ model_dir/model.json
```
