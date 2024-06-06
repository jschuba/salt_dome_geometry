Dataverse: Salt Domes in Texas
Dataset: Onshore Diapiric Salt Structure Contours in Texas
Author: John Andrews
Updated:  20231218


Introduction (from metadata):
  This Texas Data Repository dataset contains polylines representing the structure of diapiric salt within onshore
Texas. Source data for this dataset include structure contour maps between scales of 1:2000 and 1:24000 published
between 1925 and 2021. These source maps were interpreted from well data and geophysical measurements including
seismic, gravimetric, and magnetic. Contour intervals vary depending on source data; all represent salt
structure elevations relative to mean sea level (MSL).


Spatial Data Processing Steps (from metadata):
  Salt diapir mapping in Texas from geophysical sensing and other field measurements was conducted from 1925-2021
and published in journal articles, book chapters, guidebooks, and BEG circulars and reports. These data--maps,
tables, cross-sections, block diagrams, well control, and prose descriptions--were collected and used to generate
structure contour maps for each of the known 83 onshore salt diapirs in Texas. For the majority of domes, we simply
georeferenced a published structure contour map image and digitized the lines thereof; for others, we generated
new structure contours inferred from available data (methodological records are maintained in the "methodID" field
and the txSaltDiapStruct_methodID.csv table). Global Mapper was employed to generate these structure contour lines,
whether inferred or digitized directly from source. Finally, the data were compiled into a single geospatial
database using the python programming language and gis-specific modules including geopandas and shapely; the data
were output in shapefile and geojson formats and uploaded to the Texas Data Repository.



List of files:
   1) geospatial
      a) shapefile:  txSaltDiapStruct_v01.shp  (and .cpg, .dbf, .prj, .shx)
      b) geojson:    txSaltDiapStruct_v01.geojson
      c) metadata:   txSaltDiapStruct_v01_meta.xml
   2) tabular
      a) methods table:     txSaltDiapStruct_v01_methodID.tab
      b) data source table: txSaltDiapStruct_v01_datasrcID.txt
   3) other
      a) 0README.txt  (this readme file)
      b) txSaltDiapStruct_v01_mapImage.jpg  (map graphic depicting distribution of onshore Texas salt diapirs, jpg)
      c) txSaltDiapStruct_v01_mapImage.pdf  (map graphic depicting distribution of onshore Texas salt diapirs, pdf)



Credits:
  Diapiric salt mapping and map production were supported by the USGS's National Cooperative Geologic Mapping Program
(NCGMP) through STATEMAP award G22AC00495 (Jeffrey G. Paine, Principal Investigator), and by the Bureau of Economic
Geology's STARR funds for energy research (Lorena Moscardelli, Principal Investigator). Mapping and map compilation
was done by John R. Andrews.


We look forward to feedback from users of this data. Suggestions, corrections, and improved source data for future
iterations will be very much appreciated.
