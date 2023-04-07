from Compresser import Compresser

if __name__ == '__main__':
    #Read in the text from article1.txt
    with open('article2.txt', 'r') as f:
        text = f.read()
    
    #Create a Compresser object 
    compresser = Compresser(text)
    masked = compresser.masked_text_
    
    #Save that to a file called masked.txt
    with open('masked.txt', 'w') as f:
        f.write(masked)

    reconstructed = compresser.reconstructed_text_
    #Save that to a file called reconstructed.txt
    with open('reconstructed.txt', 'w') as f:
        f.write(reconstructed)