# Checkerboard Corner  Detection
Create dataset:
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

Our code draws inspiration from 'Learning Multi Instance Sub pixel Point Localization'. Please cite the following paper:

```
@InProceedings{Schroeter_2020_ACCV,
    author    = {Schroeter, Julien and Tuytelaars, Tinne and Sidorov, Kirill and Marshall, David},
    title     = {Learning Multi-Instance Sub-pixel Point Localization},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
```

