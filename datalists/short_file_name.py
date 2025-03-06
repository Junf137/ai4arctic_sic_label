import json


def process_filename(file):
    return f"{file[17:32]}_{file[77:80]}_prep.nc"


output = "processed_names.txt"

# datalist
with open("dataset_test_gt_embedded.json") as f:
    dataset_test_gt_embedded = json.load(f)
dataset_test_gt_embedded_processed = [process_filename(file) for file in dataset_test_gt_embedded]

with open("dataset.json") as f:
    dataset = json.load(f)
dataset_processed = [process_filename(file) for file in dataset]

with open("small_dataset.json") as f:
    small_dataset = json.load(f)

with open("test_val.json") as f:
    test_val = json.load(f)

with open("testset.json") as f:
    testset = json.load(f)
testset_processed = [process_filename(file) for file in testset]

with open("valset1.json") as f:
    valset1 = json.load(f)
valset1_processed = [process_filename(file) for file in valset1]

with open("valset2.json") as f:
    valset2 = json.load(f)
valset2_processed = [process_filename(file) for file in valset2]

print("---* dataset *---")
print("len: ", len(dataset))

print("---* dataset_test_gt_embedded *---")
print("len: ", len(dataset_test_gt_embedded))
print("subset of dataset", set(dataset_test_gt_embedded).issubset(set(dataset)))

print("---* small_dataset *---")
print("len: ", len(small_dataset))
print("subset of dataset", set(small_dataset).issubset(set(dataset)))

print("---* test_val *---")
print("len: ", len(test_val))
print("subset of dataset", set(test_val).issubset(set(dataset)))

print("---* testset *---")
print("len: ", len(testset))
print("subset of dataset", set(testset).issubset(set(dataset)))

print("---* valset1* ---")
print("len: ", len(valset1))
print("subset of dataset", set(valset1).issubset(set(dataset)))

print("---* valset2* ---")
print("len: ", len(valset2))
print("subset of dataset", set(valset2).issubset(set(dataset)))


print(
    "dataset_test_gt_embedded_processed same as testset_processed: ",
    set(dataset_test_gt_embedded_processed).issubset(set(testset_processed))
    and set(testset_processed).issubset(set(dataset_test_gt_embedded_processed)),
)

print("testset_processed & dataset_processed: ", set(testset_processed).intersection(set(dataset_processed)))

print(
    "valset1_processed same as valset2_processed: ",
    set(valset1_processed).issubset(set(valset2_processed)) and set(valset2_processed).issubset(set(valset1_processed)),
)


# data files
with open("./folders/train.json") as f:
    train = json.load(f)

with open("./folders/test.json") as f:
    test = json.load(f)


print("\n\n---* data files *---")
print("train len: ", len(train))
print("test len: ", len(test))

print(
    "test same as testset_processed: ", set(test).issubset(set(testset_processed)) and set(testset_processed).issubset(set(test))
)

print("testset_processed is subset of train: ", set(testset_processed).issubset(set(train)))
print("dataset_processed is subset of train: ", set(dataset_processed).issubset(set(train)))

print("testset_processed & dataset_processed: ", set(testset_processed).intersection(set(dataset_processed)))

print("train - dataset_processed - test: ", set(train).difference(set(dataset_processed)).difference(set(test)))
