import os
import base64
from time import sleep

image_type_map = lambda file_extention: file_extention.lower() if file_extention.lower() != 'jpg' else 'jpeg'

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def query_API(message, image_path=[], model='gpt-3.5-turbo', temperature=0, n=1):
    """Construct the message to the standard format, and call the api to solve.

    Args:
        message (str/list): The input message. 
            The method could construct the message by judging the type of the input.
        image_path (list[str]/list[list[str]]): The input images path. Default to []
            The method could construct the image_path by judging the type of the input.
        model (str, optional): The name of api called. Defaults to 'gpt-3.5-turbo'.

    Returns:
        str: The result.
    """
    
    system_prompt = 'You are a helpful assistant.'
    
    messages = []
    if isinstance(message, str):
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
            ]
    elif isinstance(message, list):
        messages = message
    else:
        print("The type of input message is wrong(neither 'str' or 'list').")
        return []

    if isinstance(image_path, list):
        if len(image_path) > 0:
            if isinstance(image_path[0], str):
                image_path = [image_path]
        images = [[encode_image(path) for path in path_list] for path_list in image_path]
        images_type = [[image_type_map(path.split('.')[-1]) for path in path_list] for path_list in image_path]
    else:
        print("The type of input message is wrong(neither 'list[str]' or 'list[list[str]]').")
        return []
    
    counter  = 0
    for idx in range(len(messages)):
        if messages[idx]["role"] != "user":
            continue
        if counter >= len(images):
            break
        text = messages[idx]['content']
        messages[idx]['content'] = [{"type":"text", "text":text}]
        messages[idx]['content'] += [{"type":"image", 
                                "image_url":{"url":f"data:image/{image_type};base64,{image}"}} 
                               for image, image_type in zip(images[counter], images_type[counter])]
        counter += 1

    result = api_call(messages, model=model, temperature=temperature, n=n)
    if isinstance(result, list):
        return result
    return [i.message.content for i in result.choices]


def api_call(messages, model='gpt-3.5-turbo', temperature=0, n=1):
    """Query the model to generate the response.

    Args:
        message (str): The prompt we build.
        model (str): The name of the api.

    Returns:
        str: The generated content of the model.
    """
    import openai
    from openai import AzureOpenAI, OpenAI
    import requests
    
    with open('api_key.txt') as f:
        api_key = f.readlines()
    if api_key == "" or api_key == None:
        print("None API KEY found!")
        exit()

    if model in ['gpt-3.5-turbo', '3.5', 'gpt3.5', 'chatgpt', '35', 'gpt35']:
        engine = '35turbo'
    elif model in ["gpt4", 'gpt-4', '4']:
        engine = 'gpt4'
    elif model in ["gpt4v", "gpt-4v", "4v"]:
        engine = 'gpt-4-vision-preview'
        #engine = 'gpt4'
    else:
        print(f"Wrong model Name: {model}.")
        exit()
    
    if engine != 'gpt-4-vision-preview':
        client = AzureOpenAI(
            api_version="2023-07-01-preview",
            azure_endpoint = "https://yfllm02.openai.azure.com/",
            api_key = api_key[0].strip()
        )
    while True:
        try:
            if engine == 'gpt-4-vision-preview':
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key[1]}"
                }
                payload = {
                    "model": "gpt-4-vision-preview",
                    "messages": messages, 
                    "temperature": temperature, 
                    "top_p": 1, 
                    "frequency_penalty": 0, 
                    "presence_penalty": 0, 
                    "n": n
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
                res = [r['message']['content'] for r in res['choices']]
            else:
                res = client.chat.completions.create(
                    model=engine,             # from YUN FU
                    # model=model,                    # from scirQA
                    messages=messages,
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=n
                )
            return res
        
        except openai.RateLimitError as e:
            err_mes = str(e)
            if "You exceeded your current quota" in err_mes:
                print("You exceeded your current quota: %s" % client.api_key)
            print('openai.error.RateLimitError\nRetrying...')
            sleep(30)
        except openai.APITimeoutError:
            print('openai.error.Timeout\nRetrying...')
            sleep(20)
        except openai.APIConnectionError:
            print('openai.error.APIConnectionError\nRetrying...')
            sleep(20)
        except openai.BadRequestError as e:
            print(e)
            print('openai.BadRequestError\n exit..')
            return [""]
        except requests.exceptions.ConnectTimeout:
            print('requests.exception.ConnectTimeout')
            sleep(20)
        except KeyError:
            print(res)
            return [res]
