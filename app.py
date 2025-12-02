# app.py

import logging
import tempfile
from typing import List

import pandas as pd
import streamlit as st

import plp_pipeline
from plp_pipeline import run_plp_pipeline
from mcp_client import SemrushMCPClient
from run_agent import extract_domain, fetch_page_content


# -------------------------------------------------------------------
# Streamlit logging handler
# -------------------------------------------------------------------

class StreamlitHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self._logs: List[str] = []

    def emit(self, record):
        msg = self.format(record)
        self._logs.append(msg)
        text = "\n".join(self._logs)
        # Simple text log; if you want scrolling, use st.text_area instead
        self.placeholder.text(text)


def main():
    st.set_page_config(page_title="PLP Pipeline UI", layout="wide")

    st.title("PLP Pipeline UI")

    # ---------------- Sidebar config ----------------
    with st.sidebar:
        st.header("Configuration")

        gcp_project_id = st.text_input(
            "GCP project ID",
            value="",
            help="Vertex AI project to bill against"
        )

        semrush_api_key = st.text_input(
            "Semrush API key",
            value="",
            type="password",
            help="Used by the Semrush MCP client"
        )

        database = st.text_input(
            "Semrush database",
            value="us",
            help="e.g. us, uk, de, fr"
        )

        gemini_model = st.text_input(
            "Gemini model",
            value="gemini-2.5-flash",
            help="Vertex AI model name passed into VertexLLM"
        )

        debug = st.checkbox("Debug logging", value=False)

    # stash model name so the patched VertexLLM can see it
    st.session_state["gemini_model_name"] = gemini_model

    # ---------------- URL input ----------------
    st.subheader("Input URLs")
    url_text = st.text_area(
        "One URL per line",
        value="",
        height=120,
        placeholder="https://www.example.com/page-1\nhttps://www.example.com/page-2",
    )

    run_button = st.button("Run pipeline")

    st.subheader("Logs")
    log_box = st.empty()

    result_placeholder = st.empty()
    download_placeholder = st.empty()

    if run_button:
        urls = [u.strip() for u in url_text.splitlines() if u.strip()]

        if not gcp_project_id:
            st.error("Please provide a GCP project ID.")
            return
        if not semrush_api_key:
            st.error("Please provide a Semrush API key.")
            return
        if not urls:
            st.error("Please provide at least one URL.")
            return

        # ------------- Logging into UI -------------
        handler = StreamlitHandler(log_box)
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        pipeline_logger = logging.getLogger("plp_pipeline")
        pipeline_logger.setLevel(logging.DEBUG if debug else logging.INFO)

        try:
            # ------------- Patch VertexLLM to use UI-selected Gemini model -------------
            original_vertex_llm = plp_pipeline.VertexLLM

            class UIConfiguredVertexLLM(original_vertex_llm):
                def __init__(
                    self,
                    project_id: str,
                    location: str = "us-central1",
                    model_name: str = "gemini-2.5-flash",
                    system_instruction=None,
                    temperature: float = 0.2,
                ):
                    chosen_model = st.session_state.get("gemini_model_name") or model_name
                    super().__init__(
                        project_id=project_id,
                        location=location,
                        model_name=chosen_model,
                        system_instruction=system_instruction,
                        temperature=temperature,
                    )

            # Monkey-patch the pipeline to use our subclass
            plp_pipeline.VertexLLM = UIConfiguredVertexLLM

            # Semrush client configured from UI
            semrush_client = SemrushMCPClient(api_key=semrush_api_key)

            frames = []

            for url in urls:
                logging.info(f"[UI] Processing URL: {url}")

                target_domain = extract_domain(url)
                title, h1, body = fetch_page_content(url)

                # Temporary CSV per URL
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    output_path = tmp.name

                run_plp_pipeline(
                    project_id=gcp_project_id,
                    semrush_mcp=semrush_client,
                    database=database,
                    target_domain=target_domain,
                    page_title=title or "",
                    page_h1=h1 or "",
                    page_body=body or "",
                    output_csv_path=output_path,
                    debug=debug,
                )

                df = pd.read_csv(output_path)
                df.insert(0, "source_url", url)
                frames.append(df)

            if frames:
                combined = pd.concat(frames, ignore_index=True)
                result_placeholder.subheader("Results")
                result_placeholder.dataframe(combined, use_container_width=True)

                csv_bytes = combined.to_csv(index=False).encode("utf-8")
                download_placeholder.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="plp_results.csv",
                    mime="text/csv",
                )
            else:
                result_placeholder.info("No results produced by the pipeline.")

        finally:
            # Restore original VertexLLM and remove handler
            try:
                plp_pipeline.VertexLLM = original_vertex_llm
            except Exception:
                pass
            root_logger.removeHandler(handler)


if __name__ == "__main__":
    main()
