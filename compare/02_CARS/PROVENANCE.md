# Paper and source provenance

- Correct title: *Key wavelengths screening using competitive adaptive
  reweighted sampling method for multivariate calibration*.
- Authors: Hongdong Li, Yizeng Liang, Qingsong Xu, Dongsheng Cao.
- Journal: Analytica Chimica Acta 648(1), 77-84 (2009).
- Correct DOI: `10.1016/j.aca.2009.06.046`; PMID `19616692`.
- Paper file: `paper_CARS_ACA_2009.pdf`, author's personal copy from the
  authors' archived Google Code project `carspls`.
- PDF: 9 physical pages including the Elsevier cover sheet, 1,326,126 bytes,
  SHA-256 `e64b37b260739672bb90f44a9f03a942aa02eea68fc03adae6aab68fd01a90ba`.
- Pages 1, 3, and 9 were rendered and visually verified; method equations,
  article pages 77-84, and the complete ending references are legible.
- The paper is a closed-copyright author copy for non-commercial research and
  education; do not redistribute it publicly.

The DOI `10.1016/j.aca.2009.08.046`, volume 691 and pages 46-56 supplied at the
start were incorrect and were not used for implementation.

The author's archived project describes the implementation as freely available.
The preserved official sources are:

- `official_source/carspls_1.0.0.tar.gz`, SHA-1
  `087a175f98121183e7f0ef351ac5eab9be7b43b4`, LGPL-2-or-later according to
  its DESCRIPTION file.
- `official_source/CARS_V1.50.rar`, SHA-1
  `893e2b4ff596278a5626eb440c498fc680bc9f87`, archived project source download.

The Python implementation was independently written from the paper and checked
against `carsplsorigin.R`, the author's full MC+EDF+ARS implementation. The
author package also contains `carspls.R`, explicitly documented as an EDF-only
simplification; that simplified algorithm is not substituted for full CARS.
