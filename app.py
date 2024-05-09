import streamlit as st
import requests
import json



def main():
    # PARAMETERS
    url = "http://localhost:5000/api/response"
    generate_context_url = "http://localhost:5000/api/generate_context"

    def generate_llama2_response(prompt_input, context: str, model_name: str):
        # Change input into a string
        string_dialogue = """Jesteś pomocnym asystentem. Nie odpowiadasz jako "Użytkownik" ani nie udajesz "Użytkownika". 
                            Jako "Asystent" odpowiadasz tylko raz. Na końcu odpowiedzi ZAWSZE mów, z którego pliku uzyskałeś informacje.
                            Odpowiedź należy przetłumaczyć na język POLSKI.
                            Pamiętaj o podaniu źródeł na końcu.
                            """
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

        input = {
                "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                "context": context,
                "model_name": model_name,
                }
        headers = {"Content-Type": "application/json"}
        output = requests.post(
                url=url, data=json.dumps(input), headers=headers, stream=True
                )

        return output

    def clear_chat_history():
        st.session_state.messages = [
                {"role": "assistant", "content": "Jak mogę Ci pomóc?"}
                ]

    def generate_contex():
        headers = {"Content-Type": "application/json"}
        requests.post(
            url=generate_context_url, data=json.dumps(context), headers=headers
        )

    # SIDEBAR
    with st.sidebar:
        st.title("🤖💬 Anegis chatbot")

        # MODELS AND PARAMETERS

        st.subheader("Models and parameters")
        selected_model = st.sidebar.selectbox(
            "Choose a Llama2 model", ["Llama3-8B", "Llama2-7B", "Llama2-13B"], key="selected_model"
        )
        if selected_model == "Llama2-7B":
            model_name = "Llama2-7B"
        elif selected_model == "Llama2-13B":
            model_name = "Llama2-13B"
        elif selected_model == "Llama3-8B":
            model_name = "Llama3-8B"

        selected_context = st.sidebar.selectbox(
            "Select context", ["TERG", "CRM", "ERP", "Medical"], key="selected_context"
        )
        
        if selected_context == "CRM":
            context = "CRM"
        elif selected_context == "ERP":
            context = "ERP"
        elif selected_context == "Medical":
            context = "Medical"
        elif selected_context == "TERG":
            context = "TERG"

        temperature = st.sidebar.slider(
            "temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01
        )
        max_length = st.sidebar.slider(
            "max_length", min_value=32, max_value=128, value=120, step=8
        )

        st.sidebar.button("Generate Context (VS)", on_click=generate_contex)
        st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    # CHATBOX
    prompt = st.chat_input("Write a message")

    if "messages" not in st.session_state.keys():
        clear_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt:
        with st.chat_message("User"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "User", "content": prompt})

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Myślę..."):
                    response = generate_llama2_response(prompt, context, model_name)
                    placeholder = st.empty()
                    full_response = ""
                    for item in response:
                        item_str = item.decode("utf-8")
                        full_response += item_str
                        placeholder.markdown(full_response + "▌")

                    placeholder.markdown(full_response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
