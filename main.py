from langchain_core.documents import Document
from langchain_postgres import PGVector
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# class PGVectorStore:
#     """Handles creation and population of a PGVector collection."""
#     def __init__(self, collection_name: str, connection: str):
#         self.collection_name = collection_name
#         self.connection = connection
#         self.embeddings = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-de",base_url = "http://ollama:11434")
#         self.vector_store = PGVector(
#             embeddings=self.embeddings,
#             collection_name=self.collection_name,
#             connection=self.connection,
#             use_jsonb=True,
#         )

class PGVectorStore:
    def __init__(self, database: str, user: str = "langchain", password: str = "langchain", 
                 host: str = "localhost", port: int = 6024, collection: str = "default_collection"):
        self.database = database
        self.collection = collection
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self.embeddings = OllamaEmbeddings(model="jina/jina-embeddings-v2-base-de",base_url = "http://ollama:11434")
        self.vector_store = self._initialize_store()

    def _initialize_store(self):
        """Initializes and returns the PGVector store."""
        return PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection,
            connection=self.connection_string,
            use_jsonb=True,
        )
    
    def add_documents(self, docs: list[Document], with_metadata: bool = False):
        """
        Adds documents to the PGVector collection.
        
        Args:
            docs (list[Document]): List of documents to add.
            with_metadata (bool): Whether to include metadata.
        """
        print("📝 Populating collection...")
        if with_metadata:
            self.vector_store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
        else:
            self.vector_store.add_documents(docs)
        print(f"✅ Collection '{self.collection}' populated successfully!")

import os
import requests
import tiktoken
import warnings
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
#from langchain_openai import ChatOpenAI
#from langchain_aws import ChatBedrock
from langchain_ollama.llms import OllamaLLM
from models.llm_models import LLMModel
from logger_config import log
from vector_db import retrieval

warnings.filterwarnings("ignore")
db_name = os.getenv("DATABASE_NAME")
user = os.getenv("DATABASE_USER")
password = os.getenv("DATABASE_PASSWORD")
host = os.getenv("DATABASE_HOST")
port = int(os.getenv("DATABASE_PORT"))

# _ = load_dotenv(find_dotenv())
# OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

FERN_UNI_OLLAMA_SERVER_URL = os.getenv("FERN_UNI_OLLAMA_SERVER_URL")

class LLMFeedback:
    """
    A class to interact with different language models and obtain feedback.

    Attributes:
        chatModel: An instance of a language model based on the specified llm_model.

    Methods:
        get_llm_feedback(llm_feedback_string, prompt, references, text_language):
            Generates feedback from the language model based on the provided input.
    """

    def __init__(self, llm_model):
        self.primary_model = llm_model
        self.fallback_models = [
            LLMModel.MIXTRAL,
            LLMModel.PHI4,
            LLMModel.LLAMA3,
        ]
        # Ensure the primary model is first in the fallback list
        if self.primary_model in self.fallback_models:
            self.fallback_models.remove(self.primary_model)
        self.fallback_models.insert(0, self.primary_model)

    def _get_llm(self, model_choice) -> OllamaLLM:
        try:
            if model_choice == LLMModel.LLAMA3:
                return OllamaLLM(
                    model="llama3.3:latest",
                    num_ctx=32768,
                    temperature=0.3,
                    base_url=FERN_UNI_OLLAMA_SERVER_URL,
                    num_predict=40000
                )
            elif model_choice == LLMModel.MIXTRAL:
                return OllamaLLM(
                    model="mixtral:latest",
                    num_ctx=32768,
                    temperature=0.3,
                    base_url=FERN_UNI_OLLAMA_SERVER_URL,
                    num_predict=40000
                )
            elif model_choice == LLMModel.PHI4:
                return OllamaLLM(
                    model="phi4:latest",
                    num_ctx=32768,
                    temperature=0.3,
                    base_url=FERN_UNI_OLLAMA_SERVER_URL,
                    keep_alive=-1,
                    num_predict=40000
                )
        except Exception as e:
            log.error(f"Failed to initialize model {model_choice}: {e}")
            return None
        

    def get_llm_feedback(self, llm_feedback_string, prompt, references, text_language):
        """
        Generates feedback from the language model based on the provided input.

        Args:
            llm_feedback_string (str): The input string for which feedback is required.
            prompt (str): The system prompt to guide the language model.
            references (str): Additional references to be included in the prompt.
            text_language (str): The language of the input text.

        Returns:
            tuple: A tuple containing the feedback, input tokens, output tokens, and total tokens.

        Raises:
            Exception: If the language model invocation fails.
        """
        #parser = StrOutputParser()
        tokenizer = tiktoken.get_encoding("gpt2")
        prompt = PromptTemplate(template=prompt, input_variables=["user_input", "references", "language"])
        full_prompt = prompt.format(user_input=llm_feedback_string, references=references, language=text_language)
        log.info(f"Full Prompt : \n - {full_prompt}")
        relevant_docs = self.get_relevant_docs(full_prompt, "dfki_docs", 6)
        log.info("-"*100)
        log.info(f"Relevant Document \n : - {relevant_docs}")
        for model_choice in self.fallback_models:
            try:
                llm = self._get_llm(model_choice)
                if not llm:
                    continue  # Skip if model creation failed

                chain = prompt | llm 
                llm_feedback = chain.invoke(
                    {
                        "user_input": llm_feedback_string,
                        "references": references,
                        "language": text_language
                    }
                )
                log.info(f"Model - {model_choice.name} : Success")
                input_tokens = len(tokenizer.encode(prompt.format(user_input=llm_feedback_string, references=references, language=text_language)))
                output_tokens = len(tokenizer.encode(llm_feedback))
                total_tokens = input_tokens + output_tokens
                return llm_feedback, input_tokens, output_tokens, total_tokens

            except Exception as e:
                log.error(f"Model {model_choice.name} failed: {e}")
                continue  # Try next fallback

        # If all models fail
        raise RuntimeError("All LLM models failed to generate a response.")

    def get_relevant_docs(self, user_input, collection_name, top_k=6):
        try:
            vector_db = retrieval.PGVectorStore(
                database=db_name, user=user, password=password,
                host=host, port=port, collection=collection_name
            )
            results = vector_db.similarity_search(user_input, k=top_k)
            formatted_results = vector_db.format_results(results)

            # Extracting values and adding enumeration
            all_texts = [
                f"Doc {i+1}:\n{list(d.values())[0]}"
                for i, d in enumerate(formatted_results)
            ]
            doc_count = len(all_texts)
            full_text = f"Total Documents Retrieved: {doc_count}\n\n" + "\n".join(all_texts)

            log.info(f"Formatted Results:\n{full_text}")
            return full_text

        except requests.RequestException as e:
            log.error(f"Error during request: {e}")
            return ""


