import os
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def main():
    try:
        # Download the image
        url = "https://images.pexels.com/photos/879109/pexels-photo-879109.jpeg"
        img_bytes = requests.get(url).content

        # Base64 encode
        encoded = base64.b64encode(img_bytes).decode("utf-8")

        # Convert to data URL format
        data_url = f"data:image/jpeg;base64,{encoded}"

        client_response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Generate a caption for this image in about 50 words."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ]
                }
            ]
        )

        print("Assistant:", client_response.choices[0].message.content)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
