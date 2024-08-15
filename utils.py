import os
import json
from datetime import datetime
from glob import glob
import streamlit as st
import streamlit.components.v1 as components
from langchain.schema import ChatMessage


def setup_streamlit():
    st.set_page_config(
        page_title="Financial Smart Assistant",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items={
            "About": "An in-app chat agent that allows users to ask questions about Financial news and economics, etc."
        },
    )
    
    st.title("Financial smart assistant")

    st.button(
    "Conclude your chat session",
    key="end_session",
    on_click=handle_end_session,
    help="After concluding your chat, you will be able to view a recap of your session.",
)


def change_button_colour(widget_label, prsd_status):
    if prsd_status:
        htmlstr = f"""
            <script>
                var elements = window.parent.document.querySelectorAll('button');
                for (var i = 0; i < elements.length; ++i) {{ 
                    if (elements[i].innerText == '{widget_label}') {{ 
                        elements[i].style.background = '#64B5F6'
                    }}
                }}
            </script>
            """
    else:
        htmlstr = f"""
            <script>
                var elements = window.parent.document.querySelectorAll('button');
                for (var i = 0; i < elements.length; ++i) {{ 
                    if (elements[i].innerText == '{widget_label}') {{ 
                        elements[i].style.background = null
                    }}
                }}
            </script>
            """
    components.html(f"{htmlstr}", height=0, width=0)


def handle_new_session():
    st.session_state["session_id"] = str(int(datetime.now().timestamp()))
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="How can I help you?")
    ]
    st.session_state["selected_chat"] = None
    st.session_state["read_only"] = False


def handle_end_session():
    if st.session_state["selected_chat"]:
        change_button_colour(
            widget_label=st.session_state["selected_chat"],
            prsd_status=False,
        )
    else:
        if len(st.session_state.messages) > 1:
            with open(
                os.path.join("history_chat", f"{st.session_state.session_id}.json"), "w"
            ) as f:
                json.dump({"data": [i.dict() for i in st.session_state.messages]}, f)
    handle_new_session()


def handle_chat_selection(display_chat, path):
    if st.session_state["selected_chat"] == display_chat:
        # If the same chat is clicked again, toggle off read-only mode
        handle_new_session()
        change_button_colour(widget_label=display_chat, prsd_status=False)
    else:
        if st.session_state["selected_chat"]:
            change_button_colour(
                widget_label=st.session_state["selected_chat"],
                prsd_status=False,
            )
        with open(path, "r") as f:
            data = json.load(f)["data"]
        st.session_state["messages"] = [ChatMessage(**i) for i in data]
        st.session_state["read_only"] = True
        st.session_state["selected_chat"] = display_chat
        st.session_state["session_id"] = display_chat
        change_button_colour(widget_label=display_chat, prsd_status=True)


def display_chat_buttons(history_chat):
    for idx, path in enumerate(history_chat):
        timestamp = int(os.path.basename(path).split(".")[0])
        display_chat = str(datetime.fromtimestamp(timestamp))

        if st.button(display_chat, key=f"button_{idx}"):
            handle_chat_selection(display_chat, path)


def load_chat_history():
    return sorted(glob("history_chat/*.json"), reverse=True)[:10]


def create_sidebar_expander():
    with st.sidebar.expander("Chat History"):
        history_chat = load_chat_history()
        display_chat_buttons(history_chat)