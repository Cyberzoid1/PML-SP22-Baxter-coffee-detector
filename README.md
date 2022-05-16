# PML-SP22-Baxter-coffee-detector
This project looks for the Keurig's screen and detects whether it has completed. This is an add on component to the Baxter-Project-Sp22
https://github.com/UniversityOfIdahoCDACS/Baxter-Project-Sp22

# Install
*** APT Dependancies***
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip python3-tk graphviz
```

***Python dependancies***
```bash
pip install -U pip
pip install -U requirements.txt
```

# Files
### convert.py
Used to convert a directory of raw images to a matching directory of images of the Keurig screen

### nn.py
Build and train the Neural Network

### nn_predict.py
Concept script to read an image, process, and use the AI model to predict it's classification
