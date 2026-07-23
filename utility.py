import streamlit as st

def check_password():
    """Returns True if the user entered the correct username and password."""

    def password_entered():
        if st.session_state["username"] == "scholar" and st.session_state["password"] == "pastoral2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        st.error("Wrong username or password")
        return False
    else:
        return True
    