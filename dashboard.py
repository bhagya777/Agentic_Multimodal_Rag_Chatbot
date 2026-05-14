import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="RAGAS Evaluation Dashboard", layout="wide")
st.title("📊 RAGAS Evaluation Dashboard")

LOG_FILE = os.path.join("ragas_results", "eval_log.jsonl")

if not os.path.exists(LOG_FILE):
    st.warning("No evaluation logs found yet. Start chatting in Chainlit to generate logs!")
else:
    # Read the JSONL file
    data = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        st.warning("Log file is empty.")
    else:
        df = pd.DataFrame(data)
        
        # Display high-level metrics
        st.subheader("Aggregate Average Scores")
        col1, col2, col3 = st.columns(3)
        
        metrics = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]
        cols = [col1, col2, col3]
        
        for metric, col in zip(metrics, cols):
            if metric in df.columns:
                avg_score = df[metric].mean()
                col.metric(label=metric.replace("_", " ").title(), value=f"{avg_score:.2f}")
        
        st.divider()
        
        # Display the raw data
        st.subheader("Evaluation History")
        
        # Reorder columns for better readability
        display_cols = ["timestamp", "question"] + [m for m in metrics if m in df.columns] + ["answer_length", "num_contexts", "eval_time_seconds"]
        display_df = df[[c for c in display_cols if c in df.columns]]
        
        st.dataframe(display_df, use_container_width=True)
        
        # Add some charts if we have enough data
        if len(df) > 1:
            st.divider()
            st.subheader("Score Trends Over Time")
            chart_data = df.set_index("timestamp")[[m for m in metrics if m in df.columns]]
            st.line_chart(chart_data)
