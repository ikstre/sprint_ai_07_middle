from openai import OpenAI

client = OpenAI()

def call_llm(model, prompt):
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
        max_output_tokens=300
    )

    return response.output[0].content[0].text