import openai

client = openai.OpenAI(api_key="_") 



def get_gpt_response(prompt, model="gpt-4o-mini", seed=44):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        # temperature=0,
        seed=seed
    ).choices[0].message.content
    return response

