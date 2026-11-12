"""Keep workspace controls alive when their step is not rendered."""

import streamlit as st


def persistent_key(key, *, scope, options=None, multiple=False):
    """Register a keyed input for retention across conditional views."""
    registry = dict(st.session_state.get("_workspace_widget_keys", {}))
    registry[key] = scope
    st.session_state["_workspace_widget_keys"] = registry
    if options is not None and key in st.session_state:
        value = st.session_state[key]
        if multiple:
            st.session_state[key] = [item for item in value if item in options]
        elif value not in options:
            st.session_state.pop(key)
    return key


def retain_widget_state():
    """Detach stored input values from Streamlit's hidden-widget cleanup.

    Run before declaring any widgets. Buttons and file uploaders must not be
    registered: Streamlit prohibits assigning their widget state.
    """
    for key in st.session_state.get("_workspace_widget_keys", {}):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def reset_widget_scopes(*scopes):
    """Discard controls that belong to a replaced dataset, while off screen."""
    registry = dict(st.session_state.get("_workspace_widget_keys", {}))
    for key, scope in list(registry.items()):
        if scope in scopes:
            st.session_state.pop(key, None)
            del registry[key]
    st.session_state["_workspace_widget_keys"] = registry


def persistent_file_uploader(label, *, key, **kwargs):
    """Retain an uploaded source separately from its ephemeral uploader widget."""
    uploads = dict(st.session_state.get("_workspace_uploads", {}))

    def remember():
        stored = dict(st.session_state.get("_workspace_uploads", {}))
        stored[key] = st.session_state.get(key)
        st.session_state["_workspace_uploads"] = stored

    uploaded = st.file_uploader(label, key=key, on_change=remember, **kwargs)
    if uploaded is not None:
        uploads[key] = uploaded
        st.session_state["_workspace_uploads"] = uploads
        return uploaded
    saved = st.session_state.get("_workspace_uploads", {}).get(key)
    if saved is not None:
        st.caption(f"Retained upload: {saved.name}")
        if st.button("Remove retained upload", key=f"remove_{key}"):
            uploads[key] = None
            st.session_state["_workspace_uploads"] = uploads
            st.rerun()
    return saved
