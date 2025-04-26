# --- Keep necessary imports ---
from typing import List, Dict, Optional, Any
import os
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


DEFAULT_SYNTHESIS_MODEL_PATH = "models/flan-t5-base"
SYNTHESIS_MODEL_PATH = os.getenv(
    "SYNTHESIS_MODEL_PATH", DEFAULT_SYNTHESIS_MODEL_PATH)

local_tokenizer = None
local_model = None

if not os.path.exists(SYNTHESIS_MODEL_PATH):
    logger.error(f"Synthesis model path not found: {SYNTHESIS_MODEL_PATH}")
    logger.error("Please download the 'google/flan-t5-base' model and place it in the specified path or set the SYNTHESIS_MODEL_PATH environment variable.")
else:
    try:
        logger.info(f"Loading local tokenizer from: {SYNTHESIS_MODEL_PATH}")
        local_tokenizer = AutoTokenizer.from_pretrained(
            SYNTHESIS_MODEL_PATH)

        logger.info(f"Loading local model from: {SYNTHESIS_MODEL_PATH}")

        device = "cpu"
        logger.info(f"Forcing device: {device}")
        local_model = AutoModelForSeq2SeqLM.from_pretrained(
            SYNTHESIS_MODEL_PATH).to(device)
        logger.info("Local model and tokenizer loaded successfully onto CPU.")
    
    except Exception as e:
        logger.error(
            f"Failed to load local model/tokenizer from {SYNTHESIS_MODEL_PATH}: {e}")


def synthesize_answer_with_llm(query: str, retrieved_chunks: List[Dict[str, Any]], model_name: str = "local") -> Optional[str]:
    """
    Generates a synthesized answer using a local LLM based on retrieved context.

    Args:
        query: The original user query.
        retrieved_chunks: A list of dictionaries, each containing 'text' and 'source'.
        model_name: Identifier for the model (kept for consistency, but uses local model).

    Returns:
        A synthesized answer string, or a specific error message string, or None if no context.
    """

    if not local_model or not local_tokenizer:
        logger.error(

            f"Local model/tokenizer not initialized (path: {SYNTHESIS_MODEL_PATH}). Cannot synthesize answer.")

        return f"Error: Local synthesis model not configured or failed to load from '{SYNTHESIS_MODEL_PATH}'. Cannot generate answer."

    if not retrieved_chunks:
        logger.warning("No chunks provided for synthesis.")
        return None


    context_parts = []
    source_mapping = {}
    for i, chunk_data in enumerate(retrieved_chunks):
        source_id = f"Source [{i+1}]"

        chunk_text = chunk_data.get('text', '')

        context_parts.append(f"{source_id}:\n{chunk_text[:1500]}")
        source_mapping[source_id] = chunk_data.get('source', 'Unknown')


    context_string = "\n\n".join(context_parts)

    
    if len(context_string) > 12000:
        context_string = context_string[:12000] + "\n... [Context Truncated]"
        logger.warning("Context string truncated due to length limit.")


    prompt = f'''You are an AI assistant for the EduQuery college chatbot. Your primary goal is to provide a comprehensive and detailed answer to the user's query based *only* on the provided context paragraphs (labeled Source [1], Source [2], etc.). Do not use any external knowledge or information you might have.

User Query: \\"{query}\\"

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
Answer:
'''

    # --- Calling the Local LLM ---
    try:
        logger.info(
            f"Generating answer using local model from: {SYNTHESIS_MODEL_PATH}")
        inputs = local_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(
            local_model.device)


        outputs = local_model.generate(
            **inputs,
            max_length=400,
            num_beams=4,
            early_stopping=True
        )


        synthesized_answer = local_tokenizer.decode(
            outputs[0], skip_special_tokens=True).strip()

        if not synthesized_answer:
            logger.error("Local model generated an empty response.")
            return "Error: Sorry, I couldn't generate a response at this time (empty local model response)."


        
        if "provided context does not have the answer" not in synthesized_answer and "insufficient information" not in synthesized_answer:
            source_list = "\n\n**Sources:**\n" + \
                "\n".join([f"* {sid}: {sname}" for sid,
                          sname in source_mapping.items()])
            final_answer = synthesized_answer + source_list
        else:
            final_answer = synthesized_answer

        logger.info("Successfully synthesized answer using local model.")
        return final_answer

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during local synthesis: {e}")
        logger.error(traceback.format_exc())
        return "Error: An unexpected error occurred while generating the answer locally."
