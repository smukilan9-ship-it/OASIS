# OASIS

Counts stained cells in IHC slide images.

## Download

Pick your computer. The file is large, so give it a minute.

| Your computer | Download |
|---|---|
| **Mac** (2020 or newer, Apple chip) | [OASIS-macos-arm64.zip](https://github.com/smukilan9-ship-it/OASIS/releases/latest/download/OASIS-macos-arm64.zip) |
| **Windows** | [OASIS-windows-x64.zip](https://github.com/smukilan9-ship-it/OASIS/releases/latest/download/OASIS-windows-x64.zip) |
| **Linux** | [OASIS-linux-x86_64.tar.gz](https://github.com/smukilan9-ship-it/OASIS/releases/latest/download/OASIS-linux-x86_64.tar.gz) |

Not sure which Mac you have? Click the apple in the top-left corner, then **About This
Mac**. If it says **Apple M1**, **M2**, **M3** or **M4**, use the Mac download. If it says
**Intel**, OASIS will not run on your computer.

## Install

### Mac

1. Open the downloaded file. You get an app called **OASIS**.
2. Drag it into your **Applications** folder.
3. Double-click it. A warning says the app is damaged or from an unidentified developer.
   This is expected. Nothing is wrong with the file.
4. Open **System Settings**, then **Privacy & Security**.
5. Scroll down. There is a line about OASIS being blocked. Click **Open Anyway**.
6. Double-click OASIS again and click **Open**.

You only do steps 3 to 6 once.

### Windows

1. Right-click the downloaded file and choose **Extract All**.
2. Open the folder it makes.
3. Double-click **OASIS**.
4. A blue box says "Windows protected your PC". Click **More info**, then
   **Run anyway**.

You only do step 4 once.

### Linux

1. Open a terminal in the folder you downloaded to.
2. Run these two lines:

```bash
tar -xzf OASIS-linux-x86_64.tar.gz
./OASIS/OASIS
```

If nothing opens, install the graphics packages first:

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-webkit2-4.1
```

## Using it

Open OASIS. There are four tabs down the left. Click **Need help** in the top right of
any tab for pictures of each step.

## If it does not open

- **Mac, "damaged" message keeps coming back.** Open Terminal, paste this, press Enter,
  then open the app again:

  ```bash
  xattr -dr com.apple.quarantine /Applications/OASIS.app
  ```

- **Windows, nothing happens when you double-click.** Make sure you extracted the folder
  first. Running OASIS from inside the zip does not work.

- **Anything else.** [Tell us what happened](https://github.com/smukilan9-ship-it/OASIS/issues)
  and say which computer you are on. Do not attach patient images.

## Running from the code instead

Only if you want to change the code.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
python app.py
```

Needs Python 3.10 or newer. On Windows also run `pip install pythonnet`.

## Licence

MIT. See [LICENSE](LICENSE).

The cell-detection model included with OASIS is InstanSeg, Apache-2.0. If you publish work
that used OASIS, cite InstanSeg too. See [models/NOTICE.md](models/NOTICE.md).
