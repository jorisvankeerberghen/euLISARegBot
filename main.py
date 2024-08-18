
import os
import platform
import euLISARegBot


def clear_terminal():
    current_os = platform.system()
    if current_os == "Windows":
        os.system('cls')
    else:
        os.system('clear')

# Clear the terminal screen
clear_terminal()


euLISARegBot.init_euLISARegBot()


#Interactive chatbot loop
def run_chatbot():
    print("Interactive eu-LISA Regulation Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = euLISARegBot.chatbot_response(user_input)
        print (f"euLISARegBot: {response}\n")

if __name__ == "__main__":
    run_chatbot()









