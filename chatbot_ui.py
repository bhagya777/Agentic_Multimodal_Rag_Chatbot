import chainlit as cl
import os
from rag_langgraph import query_graph
from ragas_eval import evaluate_response, format_scores

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="🤖 **Agentic RAG Research Assistant**\nAsk complex questions. I will retrieve text & visual data, run grounding checks, and cite sources step-by-step!"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    inputs = {
        "question": message.content,
        "original_question": message.content,
        "retrieved_docs": [],
        "relevant_docs": [],
        "answer": "",
        "grounded": False,
        "retry_count": 0,
        "sources_footer": "",
        "image_paths": [],
        "failure_reason": "",
        "failure_critique": "",
        "rewritten_query": "",
        "rca_history": [],
    }
    
    # Send initial status message from the main async loop
    status_msg = cl.Message(content="⏳ Searching vector database for relevant segments...")
    await status_msg.send()
    
    # Define an async function to update the UI safely
    async def update_ui(node_name, state):
        if node_name == "retrieve":
            docs = state.get("retrieved_docs", [])
            status_msg.content += f"\n✅ Retrieved {len(docs)} documents."
            status_msg.content += "\n⏳ Grading relevance of retrieved documents..."
            await status_msg.update()
            
        elif node_name == "grade_docs":
            docs = state.get("relevant_docs", [])
            status_msg.content += f"\n✅ Filtered down to {len(docs)} highly relevant documents."
            status_msg.content += "\n⏳ Generating draft answer based on filtered context..."
            await status_msg.update()
            
        elif node_name == "generate":
            status_msg.content += f"\n✅ Generated draft answer ({len(state.get('answer', ''))} characters)."
            status_msg.content += "\n⏳ Running strict hallucination grounding check..."
            await status_msg.update()
            
        elif node_name == "check_grounding":
            grounded = state.get("grounded", False)
            retry = state.get("retry_count", 0)
            if grounded:
                status_msg.content += "\n✅ Passed Hallucination Check!"
                await status_msg.update()
            else:
                status_msg.content += f"\n⚠️ Hallucination detected! (Retry count: {retry}/3)"
                status_msg.content += "\n⏳ Running Root Cause Analysis..."
                await status_msg.update()

        elif node_name == "analyze_failure":
            reason = state.get("failure_reason", "unknown")
            critique = state.get("failure_critique", "")
            emoji_map = {
                "retrieval_fail": "🔍",
                "generation_fail": "✏️",
                "unanswerable": "❌"
            }
            emoji = emoji_map.get(reason, "🔎")
            status_msg.content += f"\n{emoji} **Root Cause Analysis:** {reason}"
            status_msg.content += f"\n   → {critique}"
            if reason == "retrieval_fail":
                status_msg.content += f"\n⏳ Rewriting query and re-searching..."
            elif reason == "generation_fail":
                status_msg.content += f"\n⏳ Re-generating with stricter constraints..."
            else:
                status_msg.content += f"\n⛔ Question cannot be answered from indexed papers."
            await status_msg.update()

        elif node_name == "rewrite_query":
            new_q = state.get("question", "")
            status_msg.content += f'\n🔁 New search query: "{new_q[:80]}..."'
            status_msg.content += "\n⏳ Searching vector database with refined query..."
            await status_msg.update()

    # Natively async execution
    config = {
        "configurable": {"thread_id": "session_1"}, 
        "max_concurrency": 2  # Essential for 16GB RAM stability
    }
    
    final_state = None
    try:
        async for event in query_graph.astream(inputs, config=config):
            for node_name, state in event.items():
                final_state = state
                # Directly await the UI update
                await update_ui(node_name, state)
    except Exception as e:
        import asyncio
        if isinstance(e, asyncio.CancelledError):
            print("Graph execution cancelled by Chainlit.")
            return
        else:
            raise e
    
    if final_state:
        answer_text = final_state.get("answer", "No answer generated.")
        sources = final_state.get("sources_footer", "")
        if sources:
            answer_text += f"\n\n{sources}"
            
        # Handle images
        elements = []
        image_paths = list(set(final_state.get("image_paths", [])))
        for img_path in image_paths:
            if os.path.exists(img_path):
                elements.append(
                    cl.Image(path=img_path, name=os.path.basename(img_path), display="inline")
                )
                
        # Send the final message with embedded images
        await cl.Message(
            content=answer_text,
            elements=elements
        ).send()

        # ── RAGAS Evaluation (Silent Background Task) ──────────────
        # Collect contexts from the actual pipeline result (top 5 only to avoid RAGAS timeouts)
        docs = final_state.get("relevant_docs") or final_state.get("retrieved_docs") or []
        contexts = [
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in docs[:5]
        ]
        answer = final_state.get("answer", "")
        question = message.content

        # Run RAGAS in a background thread so it doesn't block the chatbot or hold up the UI
        def run_eval():
            evaluate_response(question, contexts, answer)

        # Dispatch as a background task
        import asyncio
        asyncio.create_task(cl.make_async(run_eval)())

    else:
        await cl.Message(content="An error occurred during processing.").send()
