# Part-Aligned Bilinear Representations for Person Re-identification

[Yumin Suh](https://cv.snu.ac.kr/~ysuh), 
[Jingdong Wang](https://jingdongwang2017.github.io/), 
[Siyu Tang](https://ps.is.tuebingen.mpg.de/person/stang), 
[Tao Mei](https://sites.google.com/view/tao-mei), 
and [Kyoung Mu Lee](https://cv.snu.ac.kr/~kmlee). “Part-Aligned Bilinear Representations for Person Re-Identification”, Proceedings of the European Conference on Computer Vision (ECCV), 2018.
([paper](https://cv.snu.ac.kr/publication/conf/2018/reid_eccv18.pdf))

```
@InProceedings{suh_eccv18,
author = {Yumin Suh and Jingdong Wang and Siyu Tang and Tao Mei and Kyoung Mu Lee},
title = {Part-Aligned Bilinear Representations for Person Re-Identification},
booktitle = {European Conference on Computer Vision (ECCV)},
year = {2018}
}
```

>Our paper is firstly implemented in caffe and reimplemented in pytorch. Minor details have been changed

Contact: Yumin Suh (n12345@snu.ac.kr)

## Prerequisite
- Pytorch 0.4

## Acknowledgement
- This code is modified from the original code [https://github.com/Cysu/open_reid](https://github.com/Cysu/open-reid)

## Usage

- Run `train_market1501.sh` to train a model on the Market-1501 dataset
- Run `eval_market1501.sh` to evaluate

- Run `train_dukemtmc.sh` to train a model on the DukeMTMC dataset
- Run `eval_dukemtmc.sh` to evaluate
