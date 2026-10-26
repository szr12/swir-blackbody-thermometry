# Real-object combustion hyperspectral dataset, v1.0.0

## Coverage

Ten source scene IDs: A4, baozhuangxiang, dingwan, jiujingdeng, jiujingkuai,
lazhu, mianhua, mutan, yinhuokuai and yinhuomu. Each has paired VNIR/SWIR
observations at 60, 120, 180 and 240 cm: 40 pairs, 80 cubes, 240 original files.
Scene IDs are original acquisition labels, not a material-composition guarantee.

Each observation contains `.hdr` (ENVI metadata), `.spe` (spectral cube) and
`.bmp` (acquisition-software preview). Original binary files are retained.
The release has 17,772,412,114 original bytes in ten compressed per-scene archives.
Extract all scene archives into the same directory to obtain `real_objects/fire/`.
The two channels are nominally identified by the original 400–1000 and 900–2500 nm
filenames; use the actual per-band wavelengths in the ENVI headers for analysis.

## Contents

- `real_objects_<scene>_v1.tar.gz`: 24 original files per scene, covering four
  distances and two channels; each archive contains a member checksum manifest.
- `real_object_metadata_v1.zip`: dataset description, acquisition inventory,
  file checksums, archive checksums and license information.
- `real_object_predictions_60cm_v1.zip`: frozen-model exploratory outputs for
  ten 60 cm SWIR scenes, plus diagnostic summaries and source hashes.
- `SHA256SUMS.txt` and individual `.sha256` files: integrity checks.

## Interpretation

The underlying measurements are digital-number responses, not absolutely calibrated
spectral radiance. Original header acquisition settings are retained. There is
one paired observation for each scene/distance combination. Image pixels and
spatially overlapping prediction windows are not independent acquisition repeats.

Prediction outputs come from the archived blackbody-trained 12-band regional
model. Their spatial and temporal correspondence to a reference temperature
sensor is unavailable, so RMSE/MAE against a registered reference are not supplied.
The values must not be treated as ground-truth temperature maps. Out-of-range
and negative predictions are retained; display color limits do not clip data.
Line-scanned observations are not verified instantaneous temperature fields.

## Code and related data

Code: https://github.com/szr12/swir-blackbody-thermometry
The separate blackbody record has reserved DOI 10.57760/sciencedb.016aq.
That identifier belongs to the blackbody record and is not the DOI of this dataset.
This dataset receives its own identifier from ScienceDB; no identifier is invented.

## License

Author-released measurements and metadata: Creative Commons Attribution 4.0
International, https://creativecommons.org/licenses/by/4.0/ . Software has its
own MIT and retained third-party conditions. Use the final ScienceDB record's
creator list and registered DOI when citing this dataset.

## 中文说明

本数据集独立于原黑体数据集，共10类实物场景、4种距离、2个光谱通道，
包括80个光谱立方体及配套头文件和软件预览图，共240个原始文件。
逐场景解压后统一位于real_objects/fire目录。以头文件中的真实波长、尺寸和
采集字段为准，不按文件名推断实际中心波长。
已有60cm SWIR模型输出作为探索性派生数据单独提供，没有配准的空间温度真值。
这些输出不代表已验证的实物温度测量精度，也不能作为真值训练标签。
