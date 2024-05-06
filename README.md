# HealthcareGPT

This repository houses the implementation of an AI-driven chatbot that leverages Databricks' model serving capabilities and vector search to provide precise answers to user queries related to healthcare information, hospital claims and transcriptions. The AI chatbot can be accessed by patients and as well as healthcare professionals. While patient can get access to information that applies to that individual patient, but the healthcare professional is able to access broader internal information in the hospital/health facility database. This chatbot integrates cutting-edge language models (LLama 3 70B) and vector search technology to ensure accurate and context-aware responses, enhancing user interaction and operational efficiency.



Below is the example of the project prototype,
[Chatbot Prototyoe](https://www.figma.com/proto/P4g84rpMGxKMn6hsA7na8W/healthassist?type=design&node-id=13-2&t=wXIz7wKHDNnMOpbu-1&scaling=scale-down&page-id=0%3A1&mode=design)

## Project Overview

The AI Chatbot is designed to assist users by answering questions spanning a wide range of topics including healthcare related questions, healthcare data analysis and visualization, and Databricks' API or infrastructure administration. The project integrates Databricks' vector search with large language models (LLMs) to fetch relevant documents and generate responses that are contextually relevant and technically accurate.

## Features

- **LLM Integration**: Utilizes a pre-trained large language model for understanding and generating human-like responses.
- **Vector Search**: Leverages Databricks' vector search capabilities to retrieve information from a corpus of documents dynamically.
- **Real-Time Responses**: Offers real-time processing of queries with efficient API calls.
- **Scalability**: Designed to scale effortlessly with the increase in data and user requests.
- **User-Friendly Interface**: Features a simple and intuitive interface built with Gradio, allowing users to interact with the AI seamlessly.

## Technical Architecture

1. **Language Model**: We use Databricks' Llama model for understanding queries and generating responses.
2. **Vector Search**: Implements Databricks vector search to find the most relevant documents based on the user's query.
3. **Retrieval System**: Combines the results from the vector search with the LLM to provide precise answers.
4. **Frontend**: A Gradio interface serves as the frontend, providing users a platform to input their queries and receive responses.

### Prerequisites

- Databricks workspace
- Access to Databricks vector search and model serving
- Python 3.8+
- Gradio
- Requests library

### Configuration

1. **Configure Environment Variables**:
   - `API_TOKEN`: Your Databricks API token.
   - `API_ENDPOINT`: Endpoint URL for the model serving.

2. **Install Dependencies**:
   ```bash
   pip install gradio requests


Usage
After launching the application, navigate to the Gradio web URL displayed in your terminal. Enter your query related to Databricks and receive an AI-generated response based on the retrieved documents.

Contributing
We welcome contributions to this project! Please feel free to fork the repository and submit pull requests. You can also open issues for bugs you've found or features you think would make a valuable addition to the project.

License
This project is licensed under the MIT License - see the LICENSE file for details.


