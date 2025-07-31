import os
import mimetypes
import base64
import chainlit as cl
from chainlit import on_message, on_chat_start
from langchain_core.messages import HumanMessage, AIMessage
from openai import OpenAI
from dotenv import load_dotenv

from utils.vectorstore_build import update_vectorstore_from_pdf

load_dotenv()


@on_chat_start
async def on_chat_start():
    await cl.Message(
        content="Hello, I'm Gluco-waIse bot! How can you help you today?"
    ).send()


@on_message
async def on_message(message: cl.Message):
    from gluco_wAIse.agent import graph
    import json

    try:
        # Handle file uploads (PDF or image)
        if message.elements:
            for element in message.elements:
                mime_type, _ = mimetypes.guess_type(element.path)

                # Handle PDF uploads to update vectorstore
                if mime_type == "application/pdf":
                    await cl.Message(
                        content="PDF received. Updating vectorstore..."
                    ).send()
                    update_vectorstore_from_pdf(element.path)
                    await cl.Message(
                        content="Vectorstore updated! Previous KB erased."
                    ).send()
                    return

                # Handle image uploads
                if mime_type and mime_type.startswith("image/"):
                    with open(element.path, "rb") as f:
                        image_data = f.read()
                    image_b64 = base64.b64encode(image_data).decode("utf-8")
                    data_url = f"data:{mime_type};base64,{image_b64}"

                    client = OpenAI()
                    # Load diabetes knowledge base content
                    kb_path = os.path.join(
                        os.path.dirname(__file__), "../data/kb/diabetes_kb.json"
                    )
                    with open(kb_path, "r") as kb_file:
                        diabetes_kb = json.load(kb_file)
                    kb_text = json.dumps(diabetes_kb)

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"Analyze the image and tell the user whether the nutrion values of the content displayed in the image follow the recommendations provided in this file: {kb_text}",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_url},
                                    },
                                ],
                            }
                        ],
                        max_tokens=500,
                    )
                    description = response.choices[0].message.content
                    await cl.Message(content=description).send()
                    return

        # Standard LangGraph flow
        user_msg = HumanMessage(content=message.content)
        state = {"messages": [user_msg]}
        result = graph.invoke(state)

        for msg in result["messages"]:
            content = getattr(msg, "content", None)
            if (
                isinstance(content, str)
                and content.endswith(".docx")
                and "word_outputs" in content
            ):
                await cl.Message(
                    content="Here is your generated Word document:",
                    elements=[cl.File(name="generated.docx", path=content)],
                ).send()
                break
            elif content and isinstance(msg, AIMessage):
                await cl.Message(content=content).send()

    except Exception as e:
        await cl.Message(content=f"Error while processing your request: {e}").send()
