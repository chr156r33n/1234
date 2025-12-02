# app.py

import json
import logging
import tempfile
from typing import List

import pandas as pd
import streamlit as st
from google.oauth2 import service_account
import vertexai

import plp_pipeline
from plp_pipeline import run_plp_pipeline
from mcp_client import SemrushMCPClient
from run_agent import extract_domain, fetch_page_content


# -------------------------------------------------------------------
# Streamlit-aware logging handler (no UI calls from background threads)
# -------------------------------------------------------------------

class StreamlitHandler(logging.Handler):
    def __init__(self, state_key: str = "log_messages"):
        super().__init__()
        self.state_key = state_key

    def emit(self, record):
        msg = self.format(record)
        # Only touch session_state; UI is updated in main thread
        logs = st.session_state.get(self.state_key, [])
        logs.append(msg)
        st.session_state[self.state_key] = logs


# -------------------------------------------------------------------
# Vertex init using uploaded service account JSON
# -------------------------------------------------------------------

def init_vertex_from_uploaded_credentials(project_id: str, location: str = "us-central1"):
    cred_info = st.session_state.get("gcp_credentials_info")
    if not cred_info:
        raise RuntimeError("GCP credentials JSON not loaded. Please upload it in the sidebar.")

    creds = service_account.Credentials.from_service_account_info(cred_info)
    vertexai.init(project=project_id, location=location, credentials=creds)


# -------------------------------------------------------------------
# Main Streamlit app
# -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="PLP Pipeline UI", layout="wide")

    # Ensure log storage exists
    st.session_state.setdefault("log_messages", [])
    st.session_state.setdefault("gcp_credentials_info", None)

    st.title("PLP Pipeline UI")

    # ---------------- Sidebar config ----------------
    with st.sidebar:
        st.header("Configuration")

        st.markdown("### GCP credentials")
        cred_file = st.file_uploader(
            "Upload GCP service account JSON",
            type=["json"],
            help="Service account with Vertex AI permissions",
        )

        if cred_file is not None:
            try:
                cred_info = json.load(cred_file)
                st.session_state["gcp_credentials_info"] = cred_info
                st.success("GCP credentials JSON loaded.")
            except Exception as e:
                st.session_state["gcp_credentials_info"] = None
                st.error(f"Could not read JSON: {e}")

        gcp_project_id = st.text_input(
            "GCP project ID",
            value="",
            help="Vertex AI project to bill against",
        )

        semrush_api_key = st.text_input(
            "Semrush API key",
            value="",
            type="password",
            help="Used by the Semrush MCP client",
        )

        database = st.text_input(
            "Semrush database",
            value="us",
            help="e.g. us, uk, de, fr",
        )

        gemini_model = st.text_input(
            "Gemini model",
            value="gemini-2.5-flash",
            help="Vertex AI model name passed into VertexLLM",
        )

        debug = st.checkbox("Debug logging", value=False)

    # store model name for the patched LLM class
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

    # ---------------- Logs + results placeholders ----------------
    st.subheader("Logs")
    log_box = st.empty()

    result_placeholder = st.empty()
    download_placeholder = st.empty()

    if run_button:
        # Clear previous logs
        st.session_state["log_messages"] = []

        urls = [u.strip() for u in url_text.splitlines() if u.strip()]

        # Basic validation
        if not st.session_state.get("gcp_credentials_info"):
            st.error("Please upload a valid GCP service account JSON in the sidebar.")
            return

        if not gcp_project_id:
            st.error("Please provide a GCP project ID.")
            return

        if not semrush_api_key:
            st.error("Please provide a Semrush API key.")
            return

        if not urls:
            st.error("Please provide at least one URL.")
            return

        st.session_state["gcp_project_id"] = gcp_project_id

        # ------------- Logging into UI (via session_state) -------------
        handler = StreamlitHandler(state_key="log_messages")
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        pipeline_logger = logging.getLogger("plp_pipeline")
        pipeline_logger.setLevel(logging.DEBUG if debug else logging.INFO)

        # Patch VertexLLM to inject our Vertex init + model name from UI
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
                # Init Vertex with uploaded SA JSON
                init_vertex_from_uploaded_credentials(project_id=project_id, location=location)

                chosen_model = st.session_state.get("gemini_model_name") or model_name

                super().__init__(
                    project_id=project_id,
                    location=location,
                    model_name=chosen_model,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )

        plp_pipeline.VertexLLM = UIConfiguredVertexLLM

        try:
            semrush_client = SemrushMCPClient(api_key=semrush_api_key)

            frames: List[pd.DataFrame] = []

            for url in urls:
                logging.info(f"[UI] Processing URL: {url}")

                target_domain = extract_domain(url)
                title, h1, body = fetch_page_content(url)

                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    output_path = tmp.name

                logging.info(f"[UI] Running PLP pipeline for {url}")
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

                # Update logs in UI after each URL
                logs = st.session_state.get("log_messages", [])
                log_box.text("\n".join(logs))

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

            # Final log refresh
            logs = st.session_state.get("log_messages", [])
            log_box.text("\n".join(logs))

        except Exception as e:
            # Throw error into UI, keep logs visible
            logging.exception("Pipeline run failed")
            st.error(f"Pipeline failed: {e}")
            logs = st.session_state.get("log_messages", [])
            log_box.text("\n".join(logs))

        finally:
            # Restore original VertexLLM and remove handler
            try:
                plp_pipeline.VertexLLM = original_vertex_llm
            except Exception:
                pass
            root_logger.removeHandler(handler)


if __name__ == "__main__":
    main()
