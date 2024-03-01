
import os
from time import sleep


def query_API(message, model='gpt-3.5-turbo', temperature=0, n=1):
    """Construct the message to the standard format, and call the api to solve.

    Args:
        message (str/list): The input message. 
            The method could construct the message by judging the type of the input.
        model (str, optional): The name of api called. Defaults to 'gpt-3.5-turbo'.

    Returns:
        str: The result.
    """
    
    system_prompt = 'You are a helpful assistant.'
    
    if type(message) == str:
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
            ]
    elif type(message) == list:
        messages = message
    
    else:
        print("The type of input message is wrong(neither 'str' or 'list').")
        return None

    result = api_call(messages, model=model, temperature=temperature)
    if type(result) == str:
        return result
    return result.choices[0].message.content
    # return [i.message.content for i in result.choices]


def api_call(messages, model='gpt-3.5-turbo', temperature=0, n=1):
    """Query the model to generate the response.

    Args:
        message (str): The prompt we build.
        model (str): The name of the api.

    Returns:
        str: The generated content of the model.
    """
    import openai
    from openai import AzureOpenAI
    
    api_key = os.getenv("YUNFU_API_KEY", "")
    if api_key == "" or api_key == None:
        print("None API KEY found!")
        exit()

    if model in ['gpt-3.5-turbo', '3.5', 'gpt3.5', 'chatgpt', '35', 'gpt35']:
        engine = '35turbo'
    elif model in ["gpt4", 'gpt-4', '4']:
        engine = 'gpt4'
    else:
        print(f"Wrong model Name: {model}.")
        exit()
    
    client = AzureOpenAI(
        api_version="2023-07-01-preview",
        azure_endpoint = "https://yfllm02.openai.azure.com/",
        api_key = api_key
    )
    while True:
        try:
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
        except openai.BadRequestError:
            print('openai.BadRequestError\nRetrying..')
            sleep(20)
            return "BadRequestError"
