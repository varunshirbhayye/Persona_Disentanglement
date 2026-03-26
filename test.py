import json

train = json.load(open("processed_data/train.json"))
val = json.load(open("processed_data/val.json"))
test = json.load(open("processed_data/test.json"))

print("Train size:", len(train))
print("Validation size:", len(val))
print("Test size:", len(test))