import uuid

from dotenv import load_dotenv
from loguru import logger

from hf_rag.agent.graph import ask, build_agent

EXAMPLE_QUESTIONS = [
    # Single-page baseline (happy path)
    "How do I fine-tune a pretrained model using the Trainer API?",
    # Multi-hop synthesis: spans PEFT setup and quantization config
    "How do I apply LoRA with PEFT and then quantize the model with bitsandbytes?",
    # Multi-hop synthesis with branching: two quantization methods, same overview page
    "What's the difference between bitsandbytes and GPTQ quantization?",
    # Out-of-scope: API reference is excluded from the corpus, expected to fail gracefully
    "What's the REST API endpoint to create a model repository programmatically?",
]


def main() -> None:
    load_dotenv()

    print("=" * 60)
    print("HF Transformers Documentation Agent")
    print("Model: OpenAI gpt-4.1-mini")
    print("=" * 60)

    logger.info("Building agent...")
    agent = build_agent()
    logger.info("Agent ready.")

    print("\nExample questions:")
    for q in EXAMPLE_QUESTIONS:
        print(f"  - {q}")
    print("\nType 'quit' to exit.\n")

    thread_id = str(uuid.uuid4())

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = ask(agent, question, thread_id=thread_id)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
