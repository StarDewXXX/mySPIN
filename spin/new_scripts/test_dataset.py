from datasets import load_dataset, load_from_disk
dataset_path = "UCLA-AGI/SPIN_iter0"
raw_dataset = load_dataset(dataset_path)
print(raw_dataset)
print(type(raw_dataset))
# train_dataset = raw_dataset['train']
def get_score(item):
    # item['weight'] = 1
    item.update({'weight':1.0})
    return item
for split in ["train","test"]:
    dataset = raw_dataset[split]
    dataset = dataset.map(get_score)
    raw_dataset[split] = dataset
    print(dataset)


print(raw_dataset['train'])
print(raw_dataset['test'])
raw_dataset.save_to_disk("./test_dataset_output/weighted-SPIN_iter0")
