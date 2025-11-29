# Record and Transcribe Audio Using Ctranslate2!
1. Transcribe voice to clipboard.
2. Paste into your favorite LLM.
3. Save time.



## Requirments
1) [Python 3.11](https://www.python.org/downloads/release/python-3119/) or [Python 3.12](https://www.python.org/downloads/release/python-31210/)
2) [Git](https://git-scm.com/downloads)
3) [git-lfs](https://git-lfs.com/)
4) Windows
  > I am open to Linux and MacOS support but would need someone to help me test it.

# Installation

### Step 1
Download the latest release (i.e. the .zip file) and extract its contents to your computer.  Then navigate to the folder containing ```main.py``` and create a virtual environment.
```
python -m venv .
```
### Step 2
Activate the virtual environment.
```
.\Scripts\activate
```
### Step 3
Run the installation script.
```
python install.py
```

# Usage

### Step 1
Activate the virtual environment and start the program:
```
python main.py
```
### Step 2
Choose the Whisper model you want to use and click "Update Settings". The first time you choose a particular model it will automatically download it.<br><br>

### Step 3
Start recording...speak...stop recording, then use ```control + v``` or right-click and "paste" the transcription into whatever program you want; for example, into the browser window for ChatGPT.<br><br>

<img width="1086" height="520" alt="image" src="https://github.com/user-attachments/assets/2e5b613b-6a14-4274-9fc1-2cf8a6cde9cc" />
* Not that the transcripton is automatically copied to the clipboard.  The "Copy to Clipboard" button allows you to edit the transcription and re-copy it.
