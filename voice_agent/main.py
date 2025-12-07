import speech_recognition as sr


def main():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        recognizer.pause_threshold = 2

        print("Listening for your command...")
        audio = recognizer.listen(source)

        print("Recognizing...")
        text = recognizer.recognize_google(audio, language="en-US")
        print(f"You said: {text}")


main()
