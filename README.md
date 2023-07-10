# Checkerboard Corner  Detection

A novel end-to-end sub-pixel checkerboard corner detection method.

```python
create_dataset.py
```

The examples of background and texture images required for synthesizing data are shown in the folder data/creat_dataset. Please add your data.

Train model:

```python
python trian.py
```
Test your image:

```python
python demo.py
```



## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
We will add information after the paper is published. For the sake of fairness in the review, we will not disclose any relevant information for the time being.
```

Our code draws inspiration from 'Learning Multi Instance Sub pixel Point Localization'. Please cite the following paper:

```
@InProceedings{Schroeter_2020_ACCV,
    author    = {Schroeter, Julien and Tuytelaars, Tinne and Sidorov, Kirill and Marshall, David},
    title     = {Learning Multi-Instance Sub-pixel Point Localization},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
```

