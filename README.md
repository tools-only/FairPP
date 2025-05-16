# Fair-PP  
Code for Neurips 2025 Datasets and Benchmarks Track submission: "Fair-PP: A Synthetic Dataset for Aligning LLM with Personalized Preferences of Social Equity"  
You can find our data here: [Fair-PP datasets](https://huggingface.co/collections/tools-o/fair-pp-6826f1f80edc145806b29a13)  

Models used in our paper:  
[Falcon3-7B-Instruct](https://huggingface.co/tiiuae/Falcon3-7B-Instruct)  
[Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)  
[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  
[Llama-SEA-LION-v3-8B-IT](https://huggingface.co/aisingapore/Llama-SEA-LION-v3-8B-IT)  
[Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)   
[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)  

structural:  
code for data generation: 1_data_generation.ipynb  
code for personalized preference collection: 2_personalized_preference.ipynb  
code for sample reweighting and analysis: 3_analysis_and_reweighting.ipynb  
code for calculate JS distance: 4_metrics.ipynb  
code for mainstream LLMs testing: llm_inference.py

Reproducing the data generation process, please make use of 1 (questions) and 2 (answers).  
If you want to obtain your own personalized preference data, you can add more persona in 2_personalized_preference.ipynb  

Otherwise, you can also directly use the dataset we provide. Using following code:  

```python
from datasets import load_dataset

dataset = load_dataset("tools-o/Fair-PP")
df = dataset["test"].to_pandas()
```
then, run code in 4_metrics.ipynb to Reproduce the experimental results of our paper.

### Term of use
The datasets and associated code are released under the CC-BY-NC-SA 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution.
