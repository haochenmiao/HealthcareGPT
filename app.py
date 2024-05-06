

import itertools
import gradio as gr
import requests
import os
from gradio.themes.utils import sizes


def respond(message, history):

    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    local_token = os.getenv('acess_token')
    local_endpoint = os.getenv('endpoint_url')

    if local_token is None or local_endpoint is None:
        return "ERROR missing env variables"

    # Add your API token to the headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {local_token}'
    }

    #prompt = list(itertools.chain.from_iterable(history))
    #prompt.append(message)
    #q = {"inputs": [prompt]}
    q = {"inputs": [message]}
    try:
        response = requests.post(
            local_endpoint, json=q, headers=headers, timeout=100)
        response_data = response.json()
        #print(response_data)
        response_data=response_data["predictions"][0]
        #print(response_data)

    except Exception as error:
        response_data = f"ERROR status_code: {type(error).__name__}"
        # + str(response.status_code) + " response:" + response.text

    # print(response.json())
    return response_data


theme = gr.themes.Soft(
    text_size=sizes.text_sm,radius_size=sizes.radius_sm, spacing_size=sizes.spacing_sm,
)


demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
    textbox=gr.Textbox(placeholder="Ask me a question",
                       container=False, scale=7),
    title="Databricks LLM RAG demo - Chat with DBRX Databricks model serving endpoint",
    description="This chatbot is a demo example for the dbdemos llm chatbot. <br>This content is provided as a LLM RAG educational example, without support. It is using DBRX, can hallucinate and should not be used as production content.<br>Please review our dbdemos license and terms for more details.",
    examples=[["What is DBRX?"],
              ["How can I start a Databricks cluster?"],
              ["What is a Databricks Cluster Policy?"],
              ["How can I track billing usage on my workspaces?"],],
    cache_examples=False,
    theme=theme,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

if __name__ == "__main__":
    demo.launch(share=True)