from PIL import Image
from VLM.client import Client

host = 'localhost'
port = 12345

def vqa(image:Image.Image, question:str):
    client = Client(host, port)
    prompt = f"USER: <image>\nQuestion: {question}\nASSISTANT:"
    ans = client.vqa([prompt], [image])
    return ans

if __name__ == '__main__':
    import json
    from tqdm import tqdm
    from PIL import Image
    from datasets import load_dataset
    client = Client(host, port)
    with open('url_map.json') as f:
        url_map = json.load(f)
    mmqa_dev = load_dataset(path='../data/mmqa.py', cache_dir='../data/mmqa_cache')['validation']

    new_dataset_split_loaded = []
    for data_item in mmqa_dev:
        data_item['table']['page_title'] = data_item['table']['title']
        new_dataset_split_loaded.append(data_item)
    mmqa_dev = new_dataset_split_loaded
    mmqa_dev = mmqa_dev[:150]

    urls = []
    for example in mmqa_dev:
        urls += example['images']['url']
    
    for url in tqdm(urls):
        image = Image.open(url_map[url]['image']['path'])
        caption = vqa(image, "Please describe this image in as much detail as possible.")
        url_map[url]['image']['new_caption'] = caption
        with open('url_map2.json', 'w') as f:
            json.dump(url_map, f)
    # with_image_map = [(k, v) for k, v in url_map.items() if 'path' in v['image'] and 'new_caption' not in v['image']]

    # for k, v in tqdm(with_image_map):
    #     image = Image.open(v['image']['path'])
    #     caption = vqa(image, "Please describe this image in as much detail as possible.")
    #     url_map[k]['image']['new_caption'] = caption
    #     with open('url_map2.json', 'w') as f:
    #         json.dump(url_map, f)
    