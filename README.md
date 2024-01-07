## Visual Odometry Pipeline




### Conda environment
Install [conda](https://docs.conda.io/projects/miniconda/en/latest/) and than run this:
```sh
conda env create --file vision.yml
```

### Run the Pipeline
```sh
# usage: main.py [-h] [--parking] [--kitty] [--malaga] [--plot] [--plot2]

# optional arguments:
#   -h, --help     show this help message and exit
#   --parking, -p  the parking dataset
#   --kitty, -k    the kitty dataset
#   --malaga, -m   the malaga dataset
#   --plot, -o     plot the results
#   --plot2, -o2   plot ground truth vs vo pipeline

# Example for plotting and selecting the parking dataset
# plot2 plots the groundtruth and vo pipeline output (only the camera poses (only kitti and parking))
python src/main.py -o -p
```

### Download datasets
To download the zipfiles for the project you can use this [script](./data/download.py). (I hope it works on windows/mac)
```sh
python data/download.py
```

## Tasks

- how to use dataclass => code example
- put init camera pose stuff into a init file
- Add PnP and write code example with the just the next frame
- add visualisation that shows which 3D points the next camera posed used
- implemented continuous pipeline (works with all frames)
- add feature matches visualisation (use heatmap if image is not working)
- KLT Tracker: Use this algorithm to track the pixels inside the matches

## Bonus 
- Boundle Adjustment
- Record Dataset
- Loop detection
- Compare Sift vs Harris vs FAST vs SURF
- (Kalman Filter)

### Done
- Feature detection and matching ( Use library, OpenCV, use efficent version?, SIFT / RANSAC ? ) $\sqrt{}$
- Setup pipeline and visualization $\sqrt{}$
- Triangulation $\sqrt{}$


## Documentation

### Major classes and important architecture decisions

