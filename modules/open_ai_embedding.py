from langchain.embeddings.openai import OpenAIEmbeddings
import os
# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_BASE"] = "https://azureopenaidcs.openai.azure.com/"
# os.environ["OPENAI_API_KEY"] = "5477144f13094f1b89e76edf66786958"
# os.environ["OPENAI_API_VERSION"] = "2"
def get_openAi_embeddings():
    embeddings = OpenAIEmbeddings()
    return embeddings