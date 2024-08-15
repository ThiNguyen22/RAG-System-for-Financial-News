import replicate
import os
import pandas as pd
import openai
import re

os.environ["REPLICATE_API_TOKEN"] = '' #API token
openai.api_key = '' #API token


def clean_text(text):

    if text is None or not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    text = re.sub(r'\*\*', '', text)  
    text = re.sub(r'""', '"', text)  
    text = re.sub(r'\#', '', text)  
    text = re.sub(r'\s+', ' ', text)  
    text = text.replace('\n', ' ')  
    text = text.strip()
    
    return text

def evaluate(model_name, actual_answer, predicted_answer):
    predicted_answer = clean_text(predicted_answer)

    prompt = f"""
    Your job is to evaluate the quality of predicted answer {actual_answer} based on the actual answer {predicted_answer} and return a label:
    - Return 1 if the predicted answer is mostly correct, meaning it contains all the key information from the actual answer, even if it includes additional context or explanations that do not contradict the key information.
    - Return -1 if the predicted answer is mostly incorrect, meaning it lacks key information or contains incorrect information compared to the actual answer.
    - Return 0 if the predicted answer states that it cannot answer the question due to insufficient information, or if it is impossible to determine if the predicted answer is correct or incorrect based on the provided information.

    Focus on the correctness of the key information in the answers. 
    Do not provide any explanation or additional text.
    Only provide the label (1, -1, or 0) on a new line by itself. 
    """
    input = {"prompt": prompt}
    if model_name in ["gpt-4o", "gpt-4-turbo"]:
        response = openai.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )
        output = response.choices[0].message.content

    else:
        response = replicate.run(model_name, input) 

    try:
        output = replicate.run(model_name, input=input)
        print("API Output:", output)
        label = int("".join(output).strip())

    except Exception as e:
        print(f"Error with model prediction from {model_name}: {e}")
        return 0
    
    return label


df = pd.read_csv('data_test.csv')
df['label'] = df.apply(lambda row: evaluate("gpt-4-turbo",row['answer'], row['prediction']), axis=1)
df.to_csv('result_gpt4_turbo.csv', index=False)

total_samples = len(df)

correct_pred = len(df[df['label'] == 1])
insufficient_pred = len(df[df['label'] == 0])
error_pred = len(df[df['label'] == -1])
print("Insufficient information: ", insufficient_pred/total_samples)
print("Error Answer: ", error_pred/total_samples)

# accuracy
accuracy = correct_pred / total_samples
print(f'Accuracy: {accuracy}')


