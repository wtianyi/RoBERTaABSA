---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from argparse import Namespace

from evaluate import main, AspectModel

import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
```

```python
args = Namespace(
    data_dir = "/your/work/space/Dataset/",
    model_name = "roberta-en",
    batch_size = 32,
    checkpoint = None
)
```

```python
result_list = []
for train_dataset in ["Restaurants", "Laptop", "Tweets"]:
    for test_dataset in ["Restaurants", "Laptop", "Tweets", "CT5K"]:
        args.dataset = train_dataset
        args.test_dataset = test_dataset
        args.checkpoint = None
        result = main(args)
        print("################################################################################")
        print(f"Trained on {train_dataset}, tested on {test_dataset}")
        print(result)
        result_list.append({
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "accuracy": result["AccuracyMetric"]["acc"],
            "precision": result["ClassifyFPreRecMetric"]["pre"],
            "recall": result["ClassifyFPreRecMetric"]["rec"],
            "f1-macro": result["ClassifyFPreRecMetric"]["f"],
        })
result_df = pd.DataFrame(result_list)
```

```python
result_df
```

```python
tmp = result_df.pivot(index="train_dataset", columns="test_dataset", values="f1-macro")
sns.heatmap(data=tmp)
plt.title("F1 score (macro)")
```
