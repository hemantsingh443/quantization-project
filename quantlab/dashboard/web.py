"""
Streamlit Web Dashboard for QuantLab.
"""

import sys
from pathlib import Path

# Add parent directory to path for running directly with streamlit
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
from typing import Optional

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def check_dependencies():
    """Check if Streamlit is available."""
    if not STREAMLIT_AVAILABLE:
        raise ImportError(
            "Streamlit is required for the web dashboard. "
            "Install with: pip install streamlit plotly pandas"
        )


def create_app():
    """Create the Streamlit dashboard application."""
    check_dependencies()
    
    from quantlab.core import QuantLab
    from quantlab.storage.experiment import Comparison
    
    # Page config
    st.set_page_config(
        page_title="QuantLab Dashboard",
        page_icon="⚡",
        layout="wide",
    )
    
    # Initialize session state
    if "lab" not in st.session_state:
        st.session_state.lab = QuantLab()
    
    lab = st.session_state.lab
    
    # Sidebar
    st.sidebar.title("⚡ QuantLab")
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🔬 New Experiment", "💬 Chat", "📈 Compare", "⚙️ Settings"]
    )
    
    if page == "📊 Dashboard":
        render_dashboard(lab)
    elif page == "🔬 New Experiment":
        render_new_experiment(lab)
    elif page == "💬 Chat":
        render_chat(lab)
    elif page == "📈 Compare":
        render_compare(lab)
    elif page == "⚙️ Settings":
        render_settings(lab)


def render_dashboard(lab):
    """Render the main dashboard view."""
    st.title("📊 Experiment Dashboard")
    
    # Statistics
    stats = lab.store.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Experiments", stats["total_experiments"])
    with col2:
        st.metric("Unique Models", len(stats.get("by_model", {})))
    with col3:
        st.metric("Quant Methods", len(stats.get("by_quant_method", {})))
    with col4:
        completed = stats.get("by_status", {}).get("completed", 0)
        st.metric("Completed", completed)
    
    st.divider()
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        model_filter = st.text_input("Filter by model name")
    with col2:
        quant_filter = st.selectbox(
            "Filter by quantization",
            ["All"] + list(stats.get("by_quant_method", {}).keys())
        )
    
    # Load experiments
    experiments = lab.list_experiments(
        model_name=model_filter if model_filter else None,
        quant_method=quant_filter if quant_filter != "All" else None,
        limit=50,
    )
    
    if not experiments:
        st.info("No experiments found. Run some experiments first!")
        return
    
    # Convert to DataFrame
    data = []
    for exp in experiments:
        data.append({
            "ID": exp.id,
            "Model": exp.model_name.split("/")[-1],
            "Size": exp.model_size or "N/A",
            "Quantization": exp.quant_method,
            "Status": exp.status,
            "Memory (MB)": exp.metrics.get("memory_mb"),
            "Latency (ms)": exp.metrics.get("latency_mean_ms"),
            "Throughput (t/s)": exp.metrics.get("throughput_tps"),
            "Timestamp": exp.timestamp,
        })
    
    df = pd.DataFrame(data)
    
    # Display table
    st.subheader("Experiments")
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )
    
    # Visualizations
    st.subheader("Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Memory by quantization method
        if "Memory (MB)" in df.columns:
            mem_df = df[df["Memory (MB)"].notna()]
            if not mem_df.empty:
                fig = px.bar(
                    mem_df,
                    x="Quantization",
                    y="Memory (MB)",
                    color="Model",
                    title="Memory Usage by Quantization",
                    barmode="group",
                )
                st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Latency by quantization method
        if "Latency (ms)" in df.columns:
            lat_df = df[df["Latency (ms)"].notna()]
            if not lat_df.empty:
                fig = px.bar(
                    lat_df,
                    x="Quantization",
                    y="Latency (ms)",
                    color="Model",
                    title="Latency by Quantization",
                    barmode="group",
                )
                st.plotly_chart(fig, width="stretch")
    
    # Experiment details
    st.subheader("Experiment Details")
    selected_id = st.selectbox(
        "Select experiment to view details",
        [exp.id for exp in experiments],
        format_func=lambda x: f"{x} - {next((e.model_name for e in experiments if e.id == x), '')}"
    )
    
    if selected_id:
        exp = lab.get_experiment(selected_id)
        if exp:
            col1, col2 = st.columns(2)
            
            with col1:
                st.json(exp.to_dict())
            
            with col2:
                st.markdown("### Metrics")
                for key, value in sorted(exp.metrics.items()):
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        st.write(f"**{key}**: {value:.4f}" if isinstance(value, float) else f"**{key}**: {value}")


def render_new_experiment(lab):
    """Render the new experiment form."""
    st.title("🔬 New Experiment")
    
    with st.form("new_experiment"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.text_input(
                "Model Name",
                value="facebook/opt-125m",
                help="HuggingFace model name or local path"
            )
            
            quant_method = st.selectbox(
                "Quantization Method",
                ["none", "fp16", "bf16", "int8", "int4", "nf4"],
            )
        
        with col2:
            exp_name = st.text_input("Experiment Name (optional)")
            tags = st.text_input("Tags (comma-separated)")
        
        st.subheader("Benchmark Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            run_benchmark = st.checkbox("Run Benchmarks", value=True)
        with col2:
            run_eval = st.checkbox("Run Evaluation", value=False)
        with col3:
            warmup_runs = st.number_input("Warmup Runs", min_value=1, value=3)
        
        submitted = st.form_submit_button("Run Experiment", type="primary")
    
    if submitted:
        with st.spinner(f"Running experiment on {model_name}..."):
            try:
                result = lab.run_experiment(
                    model_name=model_name,
                    quantization=quant_method,
                    name=exp_name if exp_name else None,
                    tags=tags.split(",") if tags else [],
                    run_benchmark=run_benchmark,
                    run_eval=run_eval,
                )
                
                st.success(f"Experiment completed! ID: {result.id}")
                st.json(result.to_dict())
                
            except Exception as e:
                st.error(f"Experiment failed: {e}")


def render_chat(lab):
    """Render the interactive chat interface for model testing."""
    st.title("💬 Model Chat")
    st.markdown("""
    Test your quantized models interactively! This provides qualitative feedback 
    that raw metrics can't capture.
    """)
    
    # Model selection
    experiments = lab.list_experiments(status="completed", limit=20)
    
    if not experiments:
        st.info("No completed experiments available. Run an experiment first!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        exp_options = {
            f"{e.id} - {e.model_name.split('/')[-1]} ({e.quant_method})": e 
            for e in experiments
        }
        selected_key = st.selectbox(
            "Select a model to chat with",
            options=list(exp_options.keys()),
        )
        selected_exp = exp_options[selected_key] if selected_key else None
    
    with col2:
        if selected_exp:
            st.metric("Memory", f"{selected_exp.metrics.get('memory_mb', 'N/A'):.1f} MB")
    
    st.divider()
    
    # Initialize chat state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = None
        st.session_state.chat_tokenizer = None
        st.session_state.chat_model_id = None
    
    # Load model button
    if selected_exp:
        model_needs_load = (
            st.session_state.chat_model is None or 
            st.session_state.chat_model_id != selected_exp.id
        )
        
        if model_needs_load:
            if st.button("Load Model for Chat", type="primary"):
                with st.spinner(f"Loading {selected_exp.model_name}..."):
                    try:
                        # Get the quantization config
                        from quantlab.quantization import get_strategy
                        
                        strategy = get_strategy(selected_exp.quant_method)
                        
                        model, tokenizer = lab.registry.load_model(
                            model_name=selected_exp.model_name,
                            quantization_config=strategy.get_load_config() if strategy else None,
                            torch_dtype=strategy.get_torch_dtype() if strategy else None,
                        )
                        
                        st.session_state.chat_model = model
                        st.session_state.chat_tokenizer = tokenizer
                        st.session_state.chat_model_id = selected_exp.id
                        st.session_state.chat_messages = []
                        
                        st.success("Model loaded! You can now chat.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
        else:
            st.success(f"✓ Model loaded: {selected_exp.model_name}")
            
            # Generation settings
            with st.expander("Generation Settings"):
                max_tokens = st.slider("Max new tokens", 10, 200, 50)
                temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
                top_p = st.slider("Top-p", 0.1, 1.0, 0.9)
            
            # Chat interface
            st.markdown("### Chat")
            
            # Display messages
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Input
            user_input = st.chat_input("Type your message...")
            
            if user_input:
                # Add user message
                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": user_input
                })
                
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Generating..."):
                        try:
                            import torch
                            
                            model = st.session_state.chat_model
                            tokenizer = st.session_state.chat_tokenizer
                            
                            inputs = tokenizer(
                                user_input, 
                                return_tensors="pt"
                            ).to(model.device)
                            
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs,
                                    max_new_tokens=max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    do_sample=True,
                                    pad_token_id=tokenizer.eos_token_id,
                                )
                            
                            response = tokenizer.decode(
                                outputs[0][inputs.input_ids.shape[1]:],
                                skip_special_tokens=True
                            )
                            
                            st.write(response)
                            
                            st.session_state.chat_messages.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                        except Exception as e:
                            st.error(f"Generation failed: {e}")
            
            # Clear chat
            if st.button("Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()
            
            # Unload model
            if st.button("Unload Model"):
                st.session_state.chat_model = None
                st.session_state.chat_tokenizer = None
                st.session_state.chat_model_id = None
                st.session_state.chat_messages = []
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
                st.rerun()


def render_compare(lab):
    """Render the comparison view."""
    st.title("📈 Compare Experiments")
    
    experiments = lab.list_experiments(limit=50)
    
    if len(experiments) < 2:
        st.warning("Need at least 2 experiments to compare.")
        return
    
    # Select experiments
    exp_options = {f"{e.id} - {e.model_name.split('/')[-1]} ({e.quant_method})": e.id for e in experiments}
    
    selected = st.multiselect(
        "Select experiments to compare",
        options=list(exp_options.keys()),
        max_selections=10,
    )
    
    if len(selected) < 2:
        st.info("Select at least 2 experiments to compare.")
        return
    
    selected_ids = [exp_options[s] for s in selected]
    comparison = lab.compare(selected_ids)
    
    if not comparison.get("experiments"):
        st.error("Failed to load selected experiments.")
        return
    
    # Comparison table
    st.subheader("Comparison Table")
    
    # Build DataFrame
    data = []
    for i, exp_info in enumerate(comparison["experiments"]):
        row = {
            "ID": exp_info["id"],
            "Model": exp_info["model"].split("/")[-1],
            "Quantization": exp_info["quant"],
        }
        for metric, values in comparison["metrics"].items():
            if i < len(values) and values[i] is not None:
                row[metric] = values[i]
        data.append(row)
    
    df = pd.DataFrame(data)
    st.dataframe(df, width="stretch", hide_index=True)
    
    # Radar chart comparison
    st.subheader("Metric Comparison")
    
    metrics_to_plot = ["memory_mb", "latency_mean_ms", "throughput_tps"]
    available_metrics = [m for m in metrics_to_plot if m in comparison["metrics"]]
    
    if available_metrics:
        fig = go.Figure()
        
        for i, exp_info in enumerate(comparison["experiments"]):
            values = []
            for metric in available_metrics:
                val = comparison["metrics"].get(metric, [None])[i]
                values.append(val if val is not None else 0)
            
            # Normalize for radar chart
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=available_metrics,
                fill='toself',
                name=f"{exp_info['id']} ({exp_info['quant']})"
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Metric Comparison (normalized)",
        )
        
        st.plotly_chart(fig, width="stretch")


def render_settings(lab):
    """Render settings page."""
    st.title("⚙️ Settings")
    
    st.subheader("Storage")
    st.write(f"**Experiments Directory:** {lab.config.experiments_dir}")
    st.write(f"**Database Path:** {lab.config.db_path}")
    
    stats = lab.store.get_statistics()
    st.write(f"**Total Experiments:** {stats['total_experiments']}")
    
    st.divider()
    
    st.subheader("Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export All Experiments"):
            export_path = lab.config.experiments_dir / "export.json"
            lab.store.export_all(export_path)
            st.success(f"Exported to {export_path}")
    
    with col2:
        uploaded = st.file_uploader("Import Experiments", type=["json"])
        if uploaded:
            import_path = lab.config.experiments_dir / "import_temp.json"
            with open(import_path, "wb") as f:
                f.write(uploaded.read())
            count = lab.store.import_from_file(import_path)
            st.success(f"Imported {count} experiments")
    
    st.divider()
    
    st.subheader("Danger Zone")
    if st.button("Clear All Experiments", type="primary"):
        if st.checkbox("I understand this will delete all experiments"):
            # Would implement delete all here
            st.warning("Not implemented for safety. Delete experiments individually.")


def launch_dashboard(port: int = 8501):
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys
    
    dashboard_path = Path(__file__)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
    ])


# For running directly with streamlit
if __name__ == "__main__" or STREAMLIT_AVAILABLE:
    if STREAMLIT_AVAILABLE:
        create_app()
