from datasets import load_dataset
x = "filtered_math.json"
ds = load_dataset("json", data_files=x, split='train', field='instances')
ds.push_to_hub("1231czx/math2")
