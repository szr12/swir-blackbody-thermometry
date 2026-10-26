# Data access

The blackbody record and the ten-scene real-object dataset are separate deposits.
The blackbody record has reserved DOI `10.57760/sciencedb.016aq`; the real-object
deposit will receive its own identifier. A reserved identifier is not evidence
of public availability. Each record must pass upload verification and repository
curation before it is advertised as a public download.

The existing blackbody collection contains 802 paired VNIR/SWIR scans under two
illumination conditions at 60 cm, with blackbody setpoints spanning 50–450 °C.
The extension contains 10 scene types, four distances (60/120/180/240 cm), two
spectral channels, 80 cubes and 240 original files including ENVI headers and previews.
Only the 60 cm SWIR subset has the reported real-object model-transfer outputs.

The small background-subtracted 401×264 benchmark matrix, uncorrected paired
matrix, scan metadata and scan-level prediction vectors are included here for
immediate numerical verification. Raw cubes are distributed through the data
repository, rather than Git history. Do not use the real-object maps as reference
temperature labels; scan-time and pixel-level registration to a reference sensor
is unavailable. Preview BMP files are acquisition-software images.
