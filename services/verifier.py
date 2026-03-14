import os
import json
from openai import OpenAI

from core.config import REASONING_MODEL, logger

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama"
)

VERIFIER_PROMPT = """You are a Fact-Checker Agent. Your job is to verify if the answer is supported by the retrieved context.

Instructions:
1. Compare the ANSWER against the CONTEXT
2. Check if all factual claims in the answer can be traced to the context
3. Flag any information in the answer that is NOT in the context as hallucination

Respond in JSON format:
{
    "is_supported": true/false - whether the answer is supported by context
    "hallucination_detected": true/false - whether the answer contains info not in context
    "feedback": "brief explanation of your verification"
}

Only mark as supported if the answer's facts match the context. Be strict."""


def verify_answer(question: str, answer: str, context: list) -> dict:
    """
    Verify if the answer is supported by the retrieved context.

    Returns:
        dict with is_supported, hallucination_detected, feedback
    """
    # Format context for verification
    context_text = "\n\n".join(
        [
            f"Source: {c.get('source', 'Unknown')}\n{c.get('content', '')[:500]}"
            for c in context[:5]
        ]
    )

    prompt = f"""Question: {question}

Answer: {answer}

Context:
{context_text}

{VERIFIER_PROMPT}"""

    try:
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )

        content = response.choices[0].message.content

        if not content:
            return {
                "is_supported": True,
                "hallucination_detected": False,
                "feedback": "Verifier returned empty response",
            }

        # Try to parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result = json.loads(content.strip())
            return {
                "is_supported": result.get("is_supported", True),
                "hallucination_detected": result.get("hallucination_detected", False),
                "feedback": result.get("feedback", "No feedback provided"),
            }
        except json.JSONDecodeError:
            # If JSON parsing fails, do simple keyword check
            return _simple_verify(question, answer, context)

    except Exception as e:
        logger.error(f"Verifier error: {e}")
        return {
            "is_supported": True,
            "hallucination_detected": False,
            "feedback": f"Verifier error: {str(e)}",
        }


def _simple_verify(question: str, answer: str, context: list) -> dict:
    """Simple fallback verification without LLM."""
    # Check if answer mentions sources
    context_sources = set()
    for c in context:
        source = c.get("source", "")
        if source and source in answer:
            context_sources.add(source)

    if not context_sources:
        return {
            "is_supported": False,
            "hallucination_detected": True,
            "feedback": "Answer does not cite any sources from context",
        }

    return {
        "is_supported": True,
        "hallucination_detected": False,
        "feedback": "Basic source citation check passed",
    }
