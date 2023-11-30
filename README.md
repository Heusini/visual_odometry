## Visual Odometry Pipeline


### Conda environment
Install [conda](https://docs.conda.io/projects/miniconda/en/latest/) and than run this:
```sh
conda env create --file vision.yml
```

### Download datasets
To download the zipfiles for the project you can use this [script](./data/download.py). (I hope it works on windows/mac)
```sh
python data/download.py
```

## Tasks

- Feature detection and matching ( Use library, OpenCV, use efficent version?, SIFT / RANSAC ? )
- Setup pipeline and visualization 
- Triangulation and PNP

## Unceartainties

- 