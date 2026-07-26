import speech_recognition as sr  
import pyttsx3  
import webbrowser
import wikipedia  
import pyjokes  
from datetime import datetime
assistant_name = "JARVIS"
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)
engine.setProperty('rate',150)
def speak(text):
    print(f"{assistant_name}:", text)
    engine.say(text)
    engine.runAndWait()
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        speak("hey boss! i am jarvis your personal ai assistant")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio)
        print("YOU SAID:",command)
        return command.lower()
    except sr.RequestError:
        print("boss please connect your internet connection")
    except sr.UnknownValueError:
        print("boss pleasse try after some time")
    except sr.WaitTimeoutError:
        print("boss you are not connected with microphone")
        return""
def ai_bot():
    speak("hey boss! i am jarvis your personal ai assistant")
    while True:
        command = listen()
        if not command:
            continue
        if "hey" in command :
            speak("hello boss")
        elif "your name" in command:
            speak("hey boss my name is jarvis and i am your personal ai assistant")
        elif "time" in command:
            now = datetime.now().strftime("%H:%M")
            speak(f"boss the time is {now}")
        elif "mathematics" in command:
            speak("a + b (Addition), a - b (Subtraction), a × b (Multiplication), a ÷ b (Division), a² (Square), a³ (Cube), √a (Square Root), ∛a (Cube Root), aⁿ (Power), n! (Factorial), π = 3.1416, e = 2.718, A = πr² (Area of Circle), C = 2πr (Circumference), c² = a² + b² (Pythagoras Theorem), sinθ = P/H, cosθ = B/H, tanθ = P/B.")
        elif "weather" in command:
            speak("boss! dharamshala has clear weather")
        elif "who's the iron man" in command:
            speak("he's my ex boss! he died fighting thanos")
        elif "branch" in command:
            speak("boss! currently you are studing in excellence technology at dharamshala branch")
        elif "home" in command:
            speak("boss you are from chakban sarotri")
        elif "family" in command:
            speak("boss there are ten members in your family")
        elif "education" in command:
            speak("boss you are graduated in bachelor of arts in geography subject in government post graduate degree college nagrota bagwan")
        elif "vehicle" in command:
            speak("boss you have one bike and one car")
        elif "setting" in command:
            speak("sorry boss you don't know the short key to opening setting")
        elif "vs code" in command:
            speak("sorry boss you don't know the short key to opening vs code")
        elif "relation" in command:
            speak("so sad boss! you are single ! you don't have any girlfriend")
        elif "contact" in command:
            speak("boss your contact number is 8544742924")
        elif "mail" in command:
            speak("boss your mail is amitsarotri@gmail.com")
        elif "laptop" in command:
            speak("boss you have acer aspire 3 laptop and intel core i5 13th generation 16gb ram and 512gb rom addede in this laptop")
        elif "routine" in command:
            speak("boss your today routine is to make new project with use of all python modules")
        elif "wikipedia" in command:
            speak("boss searching wikipedia")
            query = command.replace("wikipedia","").strip()
            try:
                result = wikipedia.summary(query,sentences=2)
                speak("according to wikipedia")
                speak(result)
            except Exception:
                speak("sorry boss i don't find the file you said you had in wikipedia")
        elif "open google" in command:
            speak("got it boss")
            webbrowser.open("www.google.com")
        elif "open youtube" in command:
            speak("got it boss")
            webbrowser.open("www.youtube.com")
        elif "open github" in command:
            speak("got it boss")
            webbrowser.open("www.github.com")
        elif "open instagram" in command:
            speak("got it boss")
            webbrowser.open("www.instagram.com")
        elif "open gmail" in command:
            speak("got it boss")
            webbrowser.open("www.gmail.com")
        elif "open netflix" in command:
            speak("got it boss")
            webbrowser.open("www.netflix.com")
        elif "open microsoft" in command:
            speak("got it boss")
            webbrowser.open("www.microsoft.com")
        elif "open facebook" in command:
            speak("got it boss")
            webbrowser.open("www.facebook.com")
        elif "open whatsapp" in command:
            speak("got it boss")
            webbrowser.open("www.web.whatsapp.com")
        elif "open spotify" in command:
            speak("got it boss")
            webbrowser.open("www.spotify.com")
        elif "open twitter" in command:
            speak("got it boss")
            webbrowser.open("www.twitter.com")
        elif "open amazon" in command:
            speak("got it boss")
            webbrowser.open("www.amazon.com")
        elif "joke" in command:
            joke = pyjokes.get_joke()
            speak(joke)
        elif "exit" in command:
            speak("see you later boss")
            break
        else:
            speak("sorry boss i don't know that command")
ai_bot()
