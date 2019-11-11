# Data

This directory contains the peak databases used during the analysis phase, and some exemplar DEM with their divide tree. 

## Pre-processing peaks

The [regionShapes](./regionShapes) folder contains our 50 hand-drawn polygons that delimitate the mountainous regions shown in the paper (see section 4.2 and supplementary material). 

First, you should download [Andrew Kirmse's mountains datasets](https://github.com/akirmse/mountains) for the peak prominences and isolations. See the link for details on the datasets, download them and replace the two files in this directory: ``prominence-p100.txt`` and ``alliso-sorted.txt``. A direct download link is also provided inside of each of either file in this repository for convenience, but keep in mind that the authors of these two files are Kirmse and de Ferranti.

Next, execute the notebook `` Analysis.ipynb`` in order to create the list of peaks per region, the orometric statistics per region and the dataset that will be used in the classifier notebook.


## DEM

We provide some real DEM and their Divide tree for illustration purposes. The DEM are provided as 16 bit grayscale PNG with the elevation encoded in feet. The spatial resolution of a pixel/cell is 3 arc second (~90m) in all cases except for the three Alps and the Pyrenees DEM, which is 1 arc second (~30m). You can assume width and height of each pixel to be the same, we already took that into account when creating the DEM. 
All these elevation maps were obtained from the SRTM-3 elevation maps corrected by Jonathan de Ferranti, which you can download from his page [viewfinderpanoramas](http://viewfinderpanoramas.org/dem3.html). 

In order to compute the Divide trees, we modified the export function from [Andrew Kirmse's code](https://github.com/akirmse/mountains) to output a TXT instead of a KML. We also had to modify his input formats in order to read elevation maps of arbitrary cell sizes and dimensions. As these two modifications were quickly done as a *hack*, we do not plan on releasing our modified code, but should be pretty straightforward to do so from his source code. Alternatively, you can contact us for details or these modified source files. Note also that, since we analyzed synthetic elevation maps as well, it made no sense to input real coordinates of the DEM and therefore all our generated Divide Trees are located starting at (lat,lon) = (0,0) even for real terrains. You can check our code or supplementary material figures for the coordinates of the real locations these DEM were taken from.


## Disclaimer about region shapes and names

We did our best to select meaningful mountainous regions based on a world relief map and our personal knowledge. Logically, there were some regions that we knew better than others, so the drawn polygon might be objectionable in terms of its shape or even its significance as a unique region. Still, we believe the main point we made about the classification (i.e. different regions have and show different orometric properties) will still be valid with any set of meaningful regions and even within a region in case there were enough samples to test it, so we encourage people to explore and compare with their own shapes and discuss their results.

Please note also that the region filename was meant to be something short and easy to locate for us. Therefore, it could not always be the most appropiate in terms of local toponymy or territorial borders. It is not our intention to imply any political meaning or cause any offense. 
