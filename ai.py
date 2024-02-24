import json
import os

import requests


def get_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {os.environ["AI_API_KEY"]}',
    }


def generate_prompt(words: list[str]) -> str:
    url = "https://api.openai.com/v1/chat/completions"

    # The note about race and ethnicity are due to some rather disturbing prompts I've gotten back, where not mentioning
    # race made the prompt hyperfocus on everyone's race, to the point where it was creepy.
    # The text is because Dall-E will sometimes try and put text in the image, which is not what we want.
    # About 250 characters is about the ideal length for an image prompt
    instructions = """
    You are feeding into an image generation model. You will be given three words, each separated by a comma.
    Return a vivid description of a dream-like scene, based on the three elements the user has provided.
    The three elements must feature prominently.
    No mentions of race, ethnicity, or text should be present in your output.
    Only return the description, as this will feed directly into the image generator.
    Limit your output to about 250 characters.
    """
    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "system",
                "content": instructions
            },
            {"role": "user", "content": ", ".join(words)},
        ],
    }
    response = requests.post(url, data=json.dumps(data), headers=get_headers())
    if response.ok:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(
            f"Failed to generate prompt: {response.status_code} {response.text}"
        )


def generate_image(prompt: str) -> str:
    url = "https://api.openai.com/v1/images/generations"
    data = {
        "prompt": f"{prompt}. You must not include any text in the image.",
        "model": "dall-e-3",
        "size": "1024x1024",
    }
    response = requests.post(url, data=json.dumps(data), headers=get_headers())
    if response.ok:
        return response.json()["data"][0]["url"]
    else:
        raise RuntimeError(
            f"Failed to generate image: {response.status_code} {response.text}"
        )
