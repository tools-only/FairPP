# Fair-PP
structural:  
code for data generation: 1_data_generation.ipynb  
code for personalized preference collection: 2_personalized_preference.ipynb  
code for sample reweighting and analysis: 3_analysis_and_reweighting.ipynb  
code for calculate JS distance: 4_metrics.ipynb  

Reproducing the data generation process, please make use of 1 (questions) and 2 (answers).  
If you want to obtain your own personalized preference data, you can add more persona in 2_personalized_preference.ipynb  

Otherwise, you can also directly use the dataset we provide [Huggingface](https://huggingface.co/collections/tools-o/fair-pp-6826f1f80edc145806b29a13). Using following code:  

```python
from datasets import load_dataset

dataset = load_dataset("tools-o/Fair-PP")
df = dataset["test"].to_pandas()
```
then, run code in 4_metrics.ipynb to Reproduce the experimental results of our paper.