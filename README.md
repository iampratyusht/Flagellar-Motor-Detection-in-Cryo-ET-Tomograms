# Locating-Bacterial-Flagellar-Motors
Finding flagellar motor centers in 3D tomograms of bacteria.

## Yaml Data Creation
```Python
    python data_utils.py --root /path/to/data --yaml_dir /path/to/output --mode train --trust 4 --neg_include
```


   ```Python
    python visualise.py --path ./data --mode random --n 10
    python visualise.py --path ./data --mode transform --n 5
    python visualise.py --path ./data --mode slices --n 10
    python visualise.py --path ./data --mode boxes --n 6 --label_path ./labels.csv```

