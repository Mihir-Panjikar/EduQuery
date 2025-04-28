from typing import List, Dict, Optional, Any
import os
import logging
import traceback
import importlib.util
import sys

logger = logging.getLogger(__name__)

# Check if ollama is installed
ollama_installed = importlib.util.find_spec("ollama") is not None
if not ollama_installed:
    logger.warning(
        "Ollama package not installed. To use Ollama models, install with: pip install ollama")
else:
    import ollama

# Add Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # Default to llama3
# Optional: Defaults to http://localhost:11434 if None
OLLAMA_HOST = os.getenv("OLLAMA_HOST")


def synthesize_answer_with_llm(query: str, retrieved_chunks: List[Dict[str, Any]], model_name: str = "ollama") -> Optional[str]:
    """
    Generates a synthesized answer using an Ollama LLM based on retrieved context.

    Args:
        query: The original user query.
        retrieved_chunks: A list of dictionaries, each containing 'text' and 'source'.
        model_name: Identifier for the model (used to specify Ollama model).

    Returns:
        A synthesized answer string, or a specific error message string, or None if no context.
    """

    if not retrieved_chunks:
        logger.warning("No chunks provided for synthesis.")
        return None  # Return None if no chunks, let the caller handle it

    # Check if ollama is installed
    if not ollama_installed:
        error_msg = "Error: Ollama package not installed. Install with: pip install ollama"
        logger.error(error_msg)
        return error_msg

    # --- Context preparation remains the same ---
    context_parts = []
    source_mapping = {}
    for i, chunk_data in enumerate(retrieved_chunks):
        source_id = f"Source [{i+1}]"
        chunk_text = chunk_data.get('text', '')
        # Limit chunk length for context window
        # Keep chunk limit reasonable
        context_parts.append(f"{source_id}:\n{chunk_text[:1500]}")
        source_mapping[source_id] = chunk_data.get('source', 'Unknown')

    context_string = "\n\n".join(context_parts)

    # Adjust overall context limit if necessary, Ollama models might have larger limits
    MAX_CONTEXT_CHARS = 8000  # Adjust as needed for llama3
    if len(context_string) > MAX_CONTEXT_CHARS:
        context_string = context_string[:MAX_CONTEXT_CHARS] + \
            "\n... [Context Truncated]"
        logger.warning(
            f"Context string truncated to {MAX_CONTEXT_CHARS} characters.")

    # --- Prompt remains the same ---
    prompt = f'''You are an AI assistant for the EduQuery college chatbot. Your primary goal is to provide a comprehensive and detailed answer to the user's query based *only* on the provided context paragraphs (labeled Source [1], Source [2], etc.). Do not use any external knowledge or information you might have.

User Query: "{query}"

Context:
---
{context_string}
---

Instructions:
1.  Carefully analyze the User Query and the provided Context paragraphs.
2.  Synthesize a thorough and elaborate answer to the query using *only* information explicitly stated in the Context. Explain the concepts clearly and provide as much relevant detail as found in the sources.
3.  If the context contains the answer, formulate it clearly and comprehensively. Cite the source(s) using the format (Source [N]) where the information was found. Combine information from multiple sources if needed, citing all relevant ones, and ensure the synthesized answer flows logically.
4.  If the context *does not* contain sufficient information relevant to the query to provide a detailed answer, respond with: "The provided context does not contain specific information to answer this query in detail." Do not attempt to guess or infer an answer.
5.  Structure the answer logically. Use bullet points or numbered lists if it helps clarity for complex information or steps.
6.  If the context is too long or complex, summarize the key points before synthesizing the answer.
7.  Do not include any disclaimers or unnecessary information in your response.
8.  Provide the answer using the sentences and phrases from the context. Do not use your own words or paraphrase.
Answer:
'''

    # --- Calling Ollama ---
    try:
        logger.info(
            f"Generating answer using Ollama model: {OLLAMA_MODEL} at host: {OLLAMA_HOST or 'default'}")

        client_args = {}
        if OLLAMA_HOST:
            client_args['host'] = OLLAMA_HOST

        client = ollama.Client(**client_args)

        try:
            # First check if the model exists and Ollama is running
            available_models = client.list()
            model_found = any(
                model['name'] == OLLAMA_MODEL for model in available_models.get('models', []))

            if not model_found:
                logger.warning(
                    f"Ollama model '{OLLAMA_MODEL}' not found. Will attempt to use it anyway.")
        except Exception as check_e:
            logger.warning(
                f"Could not verify Ollama model availability: {check_e}")
            # Continue anyway, the main call will fail with more details if there's an issue

        response = client.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])

        synthesized_answer = response['message']['content'].strip()

        if not synthesized_answer:
            logger.error("Ollama model generated an empty response.")
            return "Error: Sorry, I couldn't generate a response at this time (empty Ollama response)."

        # --- Source appending logic remains the same ---
        no_answer_phrases = [
            "provided context does not contain specific information",
            "context does not contain sufficient information",
            "insufficient information",
            "cannot answer this query",
            "information not found in the context"
        ]
        if not any(phrase in synthesized_answer.lower() for phrase in no_answer_phrases):
            source_list = "\n\n**Sources:**\n" + \
                "\n".join([f"* {sid}: {sname}" for sid,
                          sname in source_mapping.items()])
            final_answer = synthesized_answer + source_list
        else:
            # If the model explicitly says it can't answer, don't add sources
            final_answer = synthesized_answer

        logger.info("Successfully synthesized answer using Ollama model.")
        return final_answer

    except ollama.ResponseError as e:
        logger.error(
            f"Ollama API responded with an error: {e.status_code} - {e.error}")
        logger.error(traceback.format_exc())
        if e.status_code == 404:
            return f"Error: Ollama model '{OLLAMA_MODEL}' not found. Make sure it's pulled (`ollama pull {OLLAMA_MODEL}`) and Ollama is running."
        elif e.status_code == 408 or e.status_code == 504:
            return f"Error: Request to Ollama timed out. The Ollama service may be overloaded or not responding."
        elif e.status_code == 500:
            return f"Error: Ollama server error. The model might be having issues processing this query."
        return f"Error: An Ollama API error occurred ({e.status_code}). Please check if Ollama is running and accessible."
    except ConnectionError as conn_e:
        logger.error(f"Connection error to Ollama service: {conn_e}")
        logger.error(traceback.format_exc())
        return "Error: Could not connect to the Ollama service. Please ensure Ollama is running (`ollama serve`)."
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during Ollama synthesis: {e}")
        logger.error(traceback.format_exc())
        return "Error: An unexpected error occurred while generating the answer using Ollama."
