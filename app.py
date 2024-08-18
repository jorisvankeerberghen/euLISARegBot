import streamlit as st
import euLISARegBot

#Initialize the chatbot
euLISARegBot.init_euLISARegBot()

#Streamlit app layout
st.title("Interactive eu-LISA Regulation chatbot")

# Add a warning about LLMs
st.warning("⚠️ The information provided by this chatbot should be double-checked before using it in any official context. Large Language Models (LLMs) like this one are not always accurate and may generate incorrect or misleading information.")

# Display information on the regulations that have been imported
st.subheader("Imported Regulations")
st.write("""
- **Regulation (EU) 2019/817 (IO)**: Establishing a framework for interoperability between EU information systems in the field of borders and visa.
- **Regulation (EU) 2018/1240 (ETIAS)**: Establishing a European Travel Information and Authorisation System.
- **Regulation (EU) 2018/1726 (eu-LISA)**: on the European Agency for the Operational Management of Large-Scale IT Systems.
- **Regulation (EU) 2017/2226 (EES)**: Establishing an Entry/Exit System.         
""")

st.write("Need information on Regulations Related to eu-LISA? Ask your questions here!")

# Create a form
with st.form(key='chat_form'):
    user_input = st.text_input("You:")
    submit_button = st.form_submit_button(label='Send')

if user_input and submit_button:
    if user_input.lower() in ['exit', 'quit']:
        st.write("Chatbot session ended.")
    else:
        with st.spinner('Looking for the answer...'):
            response = euLISARegBot.chatbot_response(user_input)
        st.write(f" **Chatbot:** {response}")