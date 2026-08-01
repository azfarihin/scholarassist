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

    def show_login_branding():
        st.markdown("""
        <div style="background:#2D3142;padding:36px 32px;border-radius:16px;margin-bottom:24px;text-align:center;">
            <h1 style="color:#F4F1EA !important;margin:0;font-size:2.3rem;font-family:'Fraunces',serif;">ASSP Scholar Portal</h1>
            <p style="color:#C9CDD6;margin:8px 0 0 0;font-family:'Inter',sans-serif;">Sign in to access your scholarship handbook, documents, and more.</p>
        </div>
        """, unsafe_allow_html=True)

    if "password_correct" not in st.session_state:
        show_login_branding()
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        show_login_branding()
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password", on_change=password_entered)
        st.error("Wrong username or password")
        return False
    else:
        return True
