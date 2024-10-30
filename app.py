import sys
import configparser

# Azure Computer Vision
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes, OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import os
import tempfile

# Azure OpenAI
from openai import AzureOpenAI
import json

from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    ImageMessage,
)

# Config Parser
config = configparser.ConfigParser()
config.read("config.ini")

# Azure Compuer Vision Settings
vision_region = config["AzureComputerVision"]["REGION"]
vision_key = config["AzureComputerVision"]["COMPUTER_VISION_KEY"]

vision_credentials = CognitiveServicesCredentials(vision_key)
vision_client = ComputerVisionClient(
    endpoint="https://" + vision_region + ".api.cognitive.microsoft.com/",
    credentials=vision_credentials,
)

UPLOAD_FOLDER = "static"

# Dall-E 3 Settings
dalle_3_client = AzureOpenAI(
    api_version=config["AzureOpenAI"]["VERSION"],
    azure_endpoint=config["AzureOpenAI"]["ENDPOINT"],
    api_key=config["AzureOpenAI"]["OPENAI_KEY"],
)

gpt4v_client = AzureOpenAI(
    api_version=config["AzureOpenAI"]["GPT4V_VERSION"],
    azure_endpoint=config["AzureOpenAI"]["GPT4V_ENDPOINT"],
    api_key=config["AzureOpenAI"]["GPT4V_KEY"],
)

app = Flask(__name__)

channel_access_token = config["Line"]["CHANNEL_ACCESS_TOKEN"]
channel_secret = config["Line"]["CHANNEL_SECRET"]
if channel_secret is None:
    print("Specify LINE_CHANNEL_SECRET as environment variable.")
    sys.exit(1)
if channel_access_token is None:
    print("Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.")
    sys.exit(1)

handler = WebhookHandler(channel_secret)

configuration = Configuration(access_token=channel_access_token)


@app.route("/callback", methods=["POST"])
def callback():
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # parse webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def message_text(event):
    dalle3_result = openai_dalle3(event.message.text)
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    ImageMessage(
                        originalContentUrl=dalle3_result, 
                        previewImageUrl=dalle3_result
                    )
                ],
            )
        )

@handler.add(MessageEvent, message=ImageMessageContent)
def message_image(event):
    with ApiClient(configuration) as api_client:
        line_bot_blob_api = MessagingApiBlob(api_client)
        message_content = line_bot_blob_api.get_message_content(
            message_id=event.message.id
        )
        with tempfile.NamedTemporaryFile(
            dir=UPLOAD_FOLDER, prefix="", delete=False
        ) as tf:
            tf.write(message_content)
            tempfile_path = tf.name

    original_file_name = os.path.basename(tempfile_path)
    os.replace(
        UPLOAD_FOLDER + "/" + original_file_name,
        UPLOAD_FOLDER + "/" + "output.jpg",
    )
    global vision_result
    # vision_result = azure_vision()
    # vision_result = azure_vision_get_text()
    vision_result = openai_gpt4v_sdk("請詳細描述這張圖片。")
    # prompt_vision_result = "這看起來是「" + vision_result + "」。請問你想要加上什麼呢？"
    dalle3_result = openai_dalle3(vision_result)
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[
                    TextMessage(text=vision_result),
                    ImageMessage(
                        originalContentUrl=dalle3_result, 
                        previewImageUrl=dalle3_result
                    ),
                ],
            )
        )
    


def azure_vision():
    url = config["Deploy"]["WEBSITE"] + "/static/" + "output.jpg"
    language = "zh"
    max_descriptions = 3
    analysis = vision_client.describe_image(url, max_descriptions, language)
    for caption in analysis.captions:
        print(caption.text)
        print(caption.confidence)
    return analysis.captions[0].text


def openai_dalle3(user_input):
    print("vision_result:", vision_result)
    prompt_dalle3 = user_input + "在" + vision_result
    try:
        result = dalle_3_client.images.generate(
            model=config["AzureOpenAI"]["DALLE_3_DEPLOYMENT_NAME"],
            prompt=prompt_dalle3,
            n=1,
        )
        image_url = json.loads(result.model_dump_json())["data"][0]["url"]
        return image_url
    except Exception as e:
        print("Error:", e)
        return config["Deploy"]["WEBSITE"] + "/static/stop.png"


def azure_vision_get_text():
    url = config["Deploy"]["WEBSITE"] + "/static/" + "output.jpg"
    raw = True
    numberOfCharsInOperationId = 36

    # SDK call
    rawHttpResponse = vision_client.read(url, language="zh", raw=raw)

    # Get ID from returned headers
    operationLocation = rawHttpResponse.headers["Operation-Location"]
    idLocation = len(operationLocation) - numberOfCharsInOperationId
    operationId = operationLocation[idLocation:]

    # SDK call
    result = vision_client.get_read_result(operationId)

    # USe while loop to check the status of the operation
    while result.status in [
        OperationStatusCodes.not_started,
        OperationStatusCodes.running,
    ]:
        result = vision_client.get_read_result(operationId)
        print("Waiting for result : ", result)

    # Get data
    # if result.status == OperationStatusCodes.succeeded:
    print(result.status)
    return_text = ""
    for line in result.analyze_result.read_results[0].lines:
        print(line.text)
        print(line.bounding_box)
        if return_text != "":
            return_text = return_text + "," + line.text
        else:
            return_text += line.text
    # return result.analyze_result.read_results[0].lines[0].text
    return return_text

def openai_gpt4v_sdk(message_content):
    user_image_url = f"{config['Deploy']['WEBSITE']}/static/output.jpg"
    message_text = [
        {
            "role": "system",
            "content": "你觀察入微，擅長從圖像與圖表中找到資訊。使用繁體中文回答。",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message_content,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": user_image_url},
                },
            ],
        },
    ]

    try:
        response = gpt4v_client.chat.completions.create(
            model=config["AzureOpenAI"]["GPT4V_DEPLOYMENT_NAME"],
            messages=message_text,
            max_tokens=800,
            top_p=0.95,
        )
        print(response)
        return response.choices[0].message.content
    except Exception as error:
        print("Error:", error)
        return "系統異常，請再試一次。"
    
if __name__ == "__main__":
    app.run()