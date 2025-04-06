import streamlit as st

st.title("Streamlit Test")
st.write("This is a simple test to verify that Streamlit is installed correctly.")

st.header("Project Information")
st.write("Project: Guangdong Population Flow Analysis")
st.write("Description: A data analysis and visualization tool for population flow data in Guangdong, China.")

st.header("Next Steps")
st.write("1. Run the full Streamlit application using: `streamlit run app_streamlit.py`")
st.write("2. Deploy the application on Streamlit Cloud following the instructions in README_STREAMLIT.md")

if st.button("Click me!"):
    st.success("Streamlit is working correctly!")