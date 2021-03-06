# measuring-inequalities-sview

Code used for the paper: [Measuring social, environmental and health inequalities using deep learning and street imagery](https://www.nature.com/articles/s41598-019-42036-w)

## Data preperation

__1. Get labels__ `/data-prep/get_labels.py`

The initial data download for labels (for each Lower Super Output Area - LSOA) need to be completed from the corresponding websites (as described below with links). The script reads these downloaded files in, and computes the deciles from the input values that are used in training as labels. 

* [UK Census 2011](http://infusecp.mimas.ac.uk/): Detailed variable descriptions are given in the paper, and the census variable codes which can be used to replicate the analysis can be found in the script for each of the different output variables. 

* [English indices of deprivation 2015](https://www.gov.uk/government/statistics/english-indices-of-deprivation-2015)
* [Household income estimates](https://data.london.gov.uk/dataset/household-income-estimates-small-areas): Available for London only
* [ONS Postcode Directory August 2017](https://ons.maps.arcgis.com/home/item.html?id=151e4a246b91c34178a55aab047413f29b): Links  seem to be unstable, a search for ONS Postcode Directory ArcGIS gives you the up to date link for download

__2. Exctract features from images__ `/data-prep/vgg_features_extract.py`

This script uses [VGG16 pre-trained network weights](https://github.com/machrisaa/tensorflow-vgg) to extract 4096D vectors (i.e. codes) from each of the street level images used. 

__3. Create HDF5 data for training__ `/data-prep/make_hdf5.py`

Creating hdf5 files from VGG16 features and pickle files for labels to be used in training. 

Reads in: (i) features (codes) extracted from images using VGG16 (extract_features_gview.py), (ii) metadata containing labels, and input image ids

Outputs: (i) HDF5 file with features (4096D codes extracted from VGG16), (ii) labels file with corresponding variable values 

## Ordinal classification (deep learning based assignment to deciles)

`/classification/ordinal_classification_sview_tboard.py`

The assignment of each postcode to an outcome decile is an ordinal classification task, for which we used the network shown below. We used pre-trained weights of the VGG16, and only trained for the weights of the fully connected layers.

![](https://github.com/esrasuel/measuring-inequalities-sview/blob/master/classification/fig_nework_github.png)


## Aggregation

`/aggregation/get_lsoa_level_predictions.py`

Ground truth data for our analysis was only available at the Lower Super Output Area (LSOA) level for the cities we focused on. To test the performance of our networks, we needed predictions at the LSOA level and not image level. The mean continuous output value for each LSOA was computed by averaging postcode-level continuous outputs generated by the final layer of trained networks before the application of the fore-mentioned sigmoid function. The mean value computed for each LSOA was then converted to a decile category, and compared to the LSOA’s actual decile.

## Pre-trained models in Tensorflow

Will be made available soon for trained networks using London images
