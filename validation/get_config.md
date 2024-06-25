Generate Markdown file with shell command:
```Shell
jupyter nbconvert --to markdown get_config.ipynb
```


```python
import sys
sys.path.insert(0, '..')
import pymatcal
```


```python
config = pymatcal.get_config('../configs/config.yml')
print(config["dist"])
print(config['angle'])
```

    30.0
    0.0



```python

```


```python

```
