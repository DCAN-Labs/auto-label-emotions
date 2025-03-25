characters = ['Anger_Inside_Out', 'Disgust_Inside_Out', 'Fear_Inside_Out', 'Joy_Inside_Out', 'Joy_Inside_Out', 'Sadness_Inside_Out', 'Woody_Toy_Story']
emotions = ['angry', 'happy', 'sad', 'surprised']
numbers = ['0000001', '0000002', '0000003']
with open("pixar_emotions/pixar_emotions.csv", "w") as file:
    file.write('name,emotion,number' + "\n")
    for character in characters:
        for emotion in emotions:
            for number in numbers:
                line = f'{character},{emotion},{number}' + "\n"
                file.write(line)
