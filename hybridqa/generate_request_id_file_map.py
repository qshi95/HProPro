import json
import os
from tqdm import tqdm

filename = 'request_id_file_mapping.json'
resource_path = './data/WikiTables-WithLinks/'
request_path = os.path.join(resource_path, 'request_tok')
all_files = os.listdir(request_path)


mapping_dict = {}
for file in tqdm(all_files):
    file_path = os.path.join(request_path, file)
    keys = json.load(open(file_path, 'r')).keys()
    for key in keys:
        if key not in mapping_dict:
            mapping_dict[key] = file
        else:
            temp1 = json.load(open(os.path.join(request_path, file), 'r'))[key]
            temp2 = json.load(open(os.path.join(request_path, mapping_dict[key]), 'r'))[key]
            assert temp1 == temp2

# with open(filename, 'w') as fw:
#     json.dump(mapping_dict, fw)

with open(filename, "w", encoding="utf-8") as fw:
    json.dump(mapping_dict, fw, indent=4, ensure_ascii=False)