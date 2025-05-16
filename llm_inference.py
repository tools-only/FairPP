
import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

Progressive_Activists = 'Please act as one of Progressive Activists, you are highly-educated, urban. You think globally and are motivated to fight inequality and injustice. Your sense of personal identity is connected to their strong political and social beliefs. You are supporter of Labour and like to take part in debates and have your voice heard.'
Civic_Pragmatists = 'Pleases act as one of Civic Pragmatists, you are well-informed about issues and often have clear opinions, but your social and political beliefs are generally not central to your sense of personal identity. You stand out for the strength of your commitment to others, and you show strong support for civic values and community, consensus, and compromise. You feel exhausted by the division in politics.' 
Disengaged_Battlers = 'Pleases act as one of Disengaged Battlers, you are focused on the everyday struggle for survival. You have work, but often it is insecure or involves irregular hours. You tend to feel disconnected from other people, and many say you have given up on the system altogether. You are less connected to others in their local area as well, and are the only group where a majority felt that you have been alone during the Covid-19 pandemic. Although life is tough for you, you blame the system, not other people.'
Established_Liberals = 'Pleases act as one of Established Liberals, you are educated, comfortable, and quite wealthy, who feel at ease in your own skin – as well as the country you live in. You tend to trust the government, institutions, and those around you. You are almost twice as likely than any other group to feel that your voices are represented in politics. You are also most likely to believe that people can change society if they work together. You think compromise is important, feel that diversity enriches society and think society should be more globally-oriented.'
Loyal_Nationals = 'Pleases act as one of Loyal Nationals, you feel proud of your country and patriotic about its history and past achievements. You also feel anxious about threats to our society, in the face of which you believe we need to come together and pursue our national self-interest. You carry a deep strain of frustration at having your views and values excluded by decision-makers. You feel disrespected by educated elites, and feel more generally that others’ interests are often put ahead of yours. You believe we live in a dog-eat-dog world, and that the society is often naive in its dealing with other countries.'
Disengaged_Traditionalists = 'Pleases act as one of Disengaged Traditionalists, you value a feeling of self-reliance and take pride in a hard day’s work. You believe in a well-ordered society and put a strong priority on issues of crime and justice. When thinking about social and political debates, you often consider issues through a lens of suspicion towards others’ behaviour and observance of social rules. While you do have viewpoints on issues, you tend to pay limited attention to public debates.'
Backbone_Conservatives = 'Pleases act as one of Backbone Conservatives, you are confident of your nation’s place in the world. You are more prosperous than others. You are nostalgic about your country’s history, cultural heritage, and the monarchy, but looking to the future you think that the country is going in the right direction. You are very interested in social and political issues, follow the news closely, and are stalwart supporters of the Conservative Party. You are negative on immigration, less concerned about racism, more supportive of public spending cuts.'
total = [Progressive_Activists, Civic_Pragmatists, Disengaged_Battlers, Established_Liberals, Loyal_Nationals, Disengaged_Traditionalists, Backbone_Conservatives]

model_id = "/Llama-3.1-8B-Instruct" # modify your base model path here
model_path = 'Llama-3.1-8B-Instruct'

data = "test"
loss = 'dpo'
role = 6
running_list = ['normal', 'role-play', 'dpo', 'wdpo']

for m in running_list:
    if loss == 'sft':
        if role == 6:
            policy_id = f"/sft/{m}/policy.pt" # your model path after sft
    elif loss == 'dpo':
        if role == 6:
            policy_id = f"/dpo/{m}/policy.pt" # your model path after dpo

    model_path = f'Llama-3.2-3B-Instruct-{m}'
    output_path = f'./{loss}_hf_{data}_{model_path}_{role}.csv'

    if model_path != "Qwen2.5-14B-Instruct" and 'dpo' not in model_path.lower():
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
        )
    elif 'dpo' in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        model.load_state_dict(torch.load(policy_id)['state'])
    else:
        pass
    
    # load test data from huggingface
    if data == 'test':
        dataset = load_dataset("tools-o/Fair-PP")
        questions = dataset["test"].to_pandas()
        prefix = "\nDirectly select your option without explanation."
        name = 'Question'
    elif data == 'simulation':
        dataset = load_dataset("tools-o/Fair-PP-simulation")
        questions = dataset["train"].to_pandas()
        prefix = ""
        name = 'simulation_question'

    if 'dpo' in model_path.lower():
        for index, row in questions.iterrows():
            if data == 'news':
                messages = [
                    {"role": "user", "content": "Given the following article:\n" + row[name] + prefix},
                ]
            else:
                messages = [
                    {"role": "user", "content": row[name] + prefix},
                ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            questions.at[index, model_path] = response

        questions.to_csv(output_path, index=False)

    else:
        for index, row in questions.iterrows():
            if data == 'news':
                if m != 'role-play':
                    messages = [
                        {"role": "user", "content": "Given the following article:\n" + row[name] + prefix},
                    ]
                elif m == 'role-play':
                    messages = [
                        {"role": "system", "content": total[role - 1]},
                        {"role": "user", "content": "Given the following article:\n" + row[name] + prefix}
                    ]
            else:
                if m != 'role-play':
                    messages = [
                        {"role": "user", "content": row[name] + prefix},
                    ]
                elif m == 'role-play':
                    messages = [
                        {"role": "system", "content": total[role - 1]},
                        {"role": "user", "content": row[name] + prefix},
                    ]

            outputs = pipeline(
                messages,
                max_new_tokens=256,
            )
            questions.at[index, model_path] = outputs[0]["generated_text"][-1]['content']

        questions.to_csv(output_path, index=False)